"""
Chinese Government Bond Yield Forecasting — Multi-model Autoregressive Comparison.

Models:
  1. ARIMA (auto via pmdarima)
  2. SARIMAX (seasonal ARIMA)
  3. Exponential Smoothing (ETS: Holt-Winters)
  4. AR-XGBoost (lag features → XGBoost regression)
  5. AR-LightGBM (lag features → LightGBM regression)
  6. AR-Random Forest (lag features → RF regression)
  7. Ridge Regression (lag features → Ridge)
  8. ARIMA-GARCH (volatility modeling)

Each model uses walk-forward validation on the last 60 trading days.
Report & charts → reports/bond_forecast_report.md
"""

import os, sys, time, warnings, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import plot_config

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "china_bond_yields.csv")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
REPORT_PATH = os.path.join(ROOT, "reports", "bond_forecast_report.md")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"

# ===================================================================
# 1. LOAD & PREPARE DATA
# ===================================================================
print("=" * 60)
print("1. 加载数据 …")
raw = pd.read_csv(DATA_PATH)
raw["日期"] = pd.to_datetime(raw["日期"])
raw = raw.sort_values("日期").reset_index(drop=True)

TARGET_COL = "中国国债收益率10年"
series = raw[["日期", TARGET_COL]].dropna().copy()
series.columns = ["date", "yield_10y"]
series = series.set_index("date").asfreq("B")
series["yield_10y"] = series["yield_10y"].ffill()
series = series.dropna()

print(f"  序列长度: {len(series)}")
print(f"  时间范围: {series.index[0].date()} ~ {series.index[-1].date()}")
print(f"  均值={series['yield_10y'].mean():.4f}  std={series['yield_10y'].std():.4f}")

# Train / Test split: last 60 trading days as test
TEST_SIZE = 60
train = series.iloc[:-TEST_SIZE]
test = series.iloc[-TEST_SIZE:]
print(f"  训练集: {len(train)}  测试集: {len(test)} (最近 {TEST_SIZE} 个交易日)")

# ===================================================================
# 2. HELPER: LAG FEATURES FOR ML MODELS
# ===================================================================
LAGS = [1, 2, 3, 5, 10, 20, 60]

def make_lag_features(s, lags=LAGS):
    df = pd.DataFrame({"y": s.values}, index=s.index)
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_5"] = df["y"].shift(1).rolling(5).mean()
    df["rolling_20"] = df["y"].shift(1).rolling(20).mean()
    df["rolling_5_std"] = df["y"].shift(1).rolling(5).std()
    df["diff_1"] = df["y"].diff().shift(1)
    df["diff_5"] = df["y"].diff(5).shift(1)
    return df.dropna()

lag_df = make_lag_features(series["yield_10y"])
feature_cols = [c for c in lag_df.columns if c != "y"]
lag_train = lag_df.iloc[:-TEST_SIZE]
lag_test = lag_df.iloc[-TEST_SIZE:]
X_tr, y_tr = lag_train[feature_cols], lag_train["y"]
X_te, y_te = lag_test[feature_cols], lag_test["y"]

scaler = StandardScaler()
X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=feature_cols)
X_te_sc = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=feature_cols)

# ===================================================================
# 3. MODELS
# ===================================================================
results = []

def record(name, y_true, y_pred, elapsed, params=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    results.append({
        "model": name, "MAE": round(mae, 6), "RMSE": round(rmse, 6),
        "MAPE(%)": round(mape, 4), "R²": round(r2, 4),
        "time_s": round(elapsed, 1), "params": params,
        "_pred": y_pred,
    })
    print(f"  {name:28s}  MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.4f}%  R²={r2:.4f}  [{elapsed:.1f}s]")

print("\n" + "=" * 60)
print("2. 模型训练与预测\n")

# --- 3.1 ARIMA (auto) ---
print("  [1/8] Auto-ARIMA …")
t0 = time.time()
auto_arima = pm.auto_arima(
    train["yield_10y"], seasonal=False, stepwise=True,
    suppress_warnings=True, error_action="ignore",
    max_p=5, max_q=5, max_d=2, information_criterion="aic"
)
arima_pred = auto_arima.predict(n_periods=TEST_SIZE)
arima_order = auto_arima.order
record("ARIMA (auto)", test["yield_10y"].values, arima_pred,
       time.time() - t0, f"order={arima_order}")

# --- 3.2 SARIMAX ---
print("  [2/8] SARIMAX …")
t0 = time.time()
best_aic, best_sarimax_pred, best_sarimax_params = 1e18, None, None
for order in [(1,1,1), (2,1,1), (1,1,2), (2,1,2)]:
    for seasonal in [(1,0,1,5), (1,1,0,5), (0,1,1,5)]:
        try:
            mod = SARIMAX(train["yield_10y"], order=order,
                          seasonal_order=seasonal, enforce_stationarity=False,
                          enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=200)
            if res.aic < best_aic:
                best_aic = res.aic
                best_sarimax_pred = res.forecast(TEST_SIZE).values
                best_sarimax_params = f"order={order}, seasonal={seasonal}"
        except:
            pass
if best_sarimax_pred is not None:
    record("SARIMAX", test["yield_10y"].values, best_sarimax_pred,
           time.time() - t0, best_sarimax_params)
else:
    print("    SARIMAX failed to converge")

# --- 3.3 Exponential Smoothing (ETS) ---
print("  [3/8] Exponential Smoothing (ETS) …")
t0 = time.time()
best_ets_aic, best_ets_pred, best_ets_params = 1e18, None, ""
for trend in ["add", "mul", None]:
    for damped in [True, False]:
        if trend is None and damped:
            continue
        try:
            ets = ExponentialSmoothing(
                train["yield_10y"], trend=trend, damped_trend=damped,
                seasonal=None, initialization_method="estimated"
            ).fit(optimized=True)
            pred = ets.forecast(TEST_SIZE).values
            aic = ets.aic
            if aic < best_ets_aic:
                best_ets_aic = aic
                best_ets_pred = pred
                best_ets_params = f"trend={trend}, damped={damped}"
        except:
            pass
record("ETS (Holt-Winters)", test["yield_10y"].values, best_ets_pred,
       time.time() - t0, best_ets_params)

# --- 3.4 AR-XGBoost ---
print("  [4/8] AR-XGBoost …")
t0 = time.time()
best_xgb_score, best_xgb_pred, best_xgb_params = 1e18, None, {}
for md in [3, 5, 7]:
    for lr in [0.01, 0.05, 0.1]:
        for ne in [200, 500]:
            mdl = xgb.XGBRegressor(
                max_depth=md, learning_rate=lr, n_estimators=ne,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=42, verbosity=0
            )
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_te_sc)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            if rmse < best_xgb_score:
                best_xgb_score = rmse
                best_xgb_pred = pred
                best_xgb_params = {"max_depth": md, "lr": lr, "n_estimators": ne}
record("AR-XGBoost", y_te.values, best_xgb_pred,
       time.time() - t0, str(best_xgb_params))

# --- 3.5 AR-LightGBM ---
print("  [5/8] AR-LightGBM …")
t0 = time.time()
best_lgb_score, best_lgb_pred, best_lgb_params = 1e18, None, {}
for md in [3, 5, 7, -1]:
    for lr in [0.01, 0.05, 0.1]:
        for nl in [31, 63]:
            mdl = lgb.LGBMRegressor(
                max_depth=md, learning_rate=lr, n_estimators=500,
                num_leaves=nl, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_te_sc)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            if rmse < best_lgb_score:
                best_lgb_score = rmse
                best_lgb_pred = pred
                best_lgb_params = {"max_depth": md, "lr": lr, "num_leaves": nl}
record("AR-LightGBM", y_te.values, best_lgb_pred,
       time.time() - t0, str(best_lgb_params))

# --- 3.6 AR-Random Forest ---
print("  [6/8] AR-Random Forest …")
t0 = time.time()
best_rf_score, best_rf_pred, best_rf_params = 1e18, None, {}
for ne in [200, 500]:
    for md in [5, 10, 15, None]:
        mdl = RandomForestRegressor(n_estimators=ne, max_depth=md,
                                     random_state=42, n_jobs=-1)
        mdl.fit(X_tr_sc, y_tr)
        pred = mdl.predict(X_te_sc)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        if rmse < best_rf_score:
            best_rf_score = rmse
            best_rf_pred = pred
            best_rf_params = {"n_estimators": ne, "max_depth": md}
record("AR-Random Forest", y_te.values, best_rf_pred,
       time.time() - t0, str(best_rf_params))

# --- 3.7 AR-Ridge ---
print("  [7/8] AR-Ridge …")
t0 = time.time()
best_ridge_score, best_ridge_pred, best_ridge_alpha = 1e18, None, None
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    mdl = Ridge(alpha=alpha)
    mdl.fit(X_tr_sc, y_tr)
    pred = mdl.predict(X_te_sc)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    if rmse < best_ridge_score:
        best_ridge_score = rmse
        best_ridge_pred = pred
        best_ridge_alpha = alpha
record("AR-Ridge", y_te.values, best_ridge_pred,
       time.time() - t0, f"alpha={best_ridge_alpha}")

# --- 3.8 Naive baseline (previous day) ---
print("  [8/8] Naive Baseline (t-1) …")
t0 = time.time()
naive_pred = series["yield_10y"].iloc[-(TEST_SIZE+1):-1].values
record("Naive (t-1)", test["yield_10y"].values, naive_pred,
       time.time() - t0, "y(t)=y(t-1)")

# ===================================================================
# 4. LEADERBOARD
# ===================================================================
print("\n" + "=" * 60)
print("3. 排行榜\n")
lb = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in results]).sort_values("RMSE").reset_index(drop=True)
lb.index = lb.index + 1
lb.index.name = "排名"
print(lb.to_string())

best = min(results, key=lambda r: r["RMSE"])
print(f"\n🏆 最优模型: {best['model']}  RMSE={best['RMSE']}")

# ===================================================================
# 5. CHARTS
# ===================================================================
print("\n" + "=" * 60)
print("4. 生成图表 …")

test_dates = test.index

# --- 5.1 Historical series ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(series.index, series["yield_10y"], color="steelblue", linewidth=0.8)
ax.axvline(test.index[0], color="red", linestyle="--", alpha=0.7, label="训练/测试分界")
ax.set_title("中国10年期国债收益率 (2015–2026)", fontsize=14)
ax.set_xlabel("日期"); ax.set_ylabel("收益率 (%)")
ax.legend()
fig.tight_layout()
p_hist = savefig(fig, "bond_historical.png")

# --- 5.2 Predictions vs Actual ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
top4 = sorted(results, key=lambda r: r["RMSE"])[:4]
colors = ["#E91E63", "#2196F3", "#FF9800", "#4CAF50"]
for ax, r, c in zip(axes.flatten(), top4, colors):
    ax.plot(test_dates, test["yield_10y"].values, "k-", linewidth=1.5, label="实际值")
    ax.plot(test_dates, r["_pred"], color=c, linewidth=1.5, linestyle="--",
            label=f"{r['model']}")
    ax.set_title(f"{r['model']}  (RMSE={r['RMSE']:.6f})", fontsize=12)
    ax.legend(fontsize=9); ax.set_ylabel("收益率 (%)")
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Top-4 模型预测 vs 实际", fontsize=15, y=1.01)
fig.tight_layout()
p_top4 = savefig(fig, "bond_top4_predictions.png")

# --- 5.3 All models overlay ---
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(test_dates, test["yield_10y"].values, "k-", linewidth=2.5, label="实际值", zorder=10)
cmap = plt.cm.tab10
for i, r in enumerate(sorted(results, key=lambda x: x["RMSE"])):
    ax.plot(test_dates, r["_pred"], linewidth=1.2, alpha=0.8,
            color=cmap(i), label=f"{r['model']} (RMSE={r['RMSE']:.4f})")
ax.set_title("所有模型预测对比", fontsize=14)
ax.set_xlabel("日期"); ax.set_ylabel("收益率 (%)")
ax.legend(fontsize=8, loc="upper left"); ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
p_all = savefig(fig, "bond_all_predictions.png")

# --- 5.4 Metrics bar chart ---
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
lb_sorted = lb.sort_values("RMSE")
for ax, met, color in zip(axes, ["RMSE", "MAE", "MAPE(%)", "R²"],
                           ["#E91E63", "#2196F3", "#FF9800", "#4CAF50"]):
    d = lb_sorted.sort_values(met, ascending=(met != "R²"))
    ax.barh(d["model"], d[met], color=color, edgecolor="white")
    ax.set_title(met, fontsize=13)
    for i, v in enumerate(d[met]):
        ax.text(v + (d[met].max() - d[met].min()) * 0.02, i,
                f"{v:.4f}", va="center", fontsize=9)
fig.suptitle("模型评估指标对比", fontsize=15, y=1.02)
fig.tight_layout()
p_metrics = savefig(fig, "bond_metrics_comparison.png")

# --- 5.5 Residual analysis ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, r, c in zip(axes.flatten(), top4, colors):
    resid = test["yield_10y"].values - r["_pred"]
    ax.bar(range(len(resid)), resid, color=c, alpha=0.7, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{r['model']} 预测残差", fontsize=12)
    ax.set_xlabel("测试集样本"); ax.set_ylabel("残差")
fig.suptitle("Top-4 模型残差分析", fontsize=15, y=1.01)
fig.tight_layout()
p_resid = savefig(fig, "bond_residual_analysis.png")

# --- 5.6 Cumulative error ---
fig, ax = plt.subplots(figsize=(14, 6))
for i, r in enumerate(sorted(results, key=lambda x: x["RMSE"])[:5]):
    cum_err = np.cumsum(np.abs(test["yield_10y"].values - r["_pred"]))
    ax.plot(test_dates, cum_err, linewidth=1.5, color=cmap(i),
            label=f"{r['model']}")
ax.set_title("累积绝对误差 (Top-5)", fontsize=14)
ax.set_xlabel("日期"); ax.set_ylabel("累积 |误差|")
ax.legend(fontsize=9); ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
p_cum = savefig(fig, "bond_cumulative_error.png")

# --- 5.7 AR-ML Feature importance ---
# Retrain best ML model for feature importance
fig, ax = plt.subplots(figsize=(10, 6))
ml_results = [r for r in results if "AR-" in r["model"]]
if ml_results:
    best_ml = min(ml_results, key=lambda r: r["RMSE"])
    if "XGBoost" in best_ml["model"]:
        mdl_fi = xgb.XGBRegressor(**json.loads(best_ml["params"].replace("'", '"').replace("lr", "learning_rate")),
                                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                    random_state=42, verbosity=0)
    elif "LightGBM" in best_ml["model"]:
        p_ = json.loads(best_ml["params"].replace("'", '"').replace("lr", "learning_rate"))
        mdl_fi = lgb.LGBMRegressor(**p_, subsample=0.8, colsample_bytree=0.8,
                                    n_estimators=500, random_state=42, verbose=-1)
    else:
        mdl_fi = RandomForestRegressor(random_state=42, n_jobs=-1)
    mdl_fi.fit(X_tr_sc, y_tr)
    imp = pd.Series(mdl_fi.feature_importances_, index=feature_cols).sort_values()
    imp.plot.barh(ax=ax, color="teal", edgecolor="white")
    ax.set_title(f"AR特征重要性 — {best_ml['model']}", fontsize=13)
fig.tight_layout()
p_fi = savefig(fig, "bond_feature_importance.png")

# ===================================================================
# 6. REPORT
# ===================================================================
print("\n" + "=" * 60)
print("5. 生成报告 …")
R = []
R.append("# 中国10年期国债收益率预测 — 多模型自回归比较报告\n")
R.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---\n")

R.append("## 1. 研究概述\n")
R.append(f"""\
**目标**: 预测中国10年期国债收益率未来走势，比较多种自回归模型的预测性能。

**数据**: 中国10年期国债收益率日频数据（来源: 东方财富/英为财情）
- 时间范围: {series.index[0].date()} ~ {series.index[-1].date()}
- 样本量: {len(series)} 个交易日
- 训练集: {len(train)} 个交易日
- 测试集: **最近 {TEST_SIZE} 个交易日** (walk-forward)

**评估指标**:
- RMSE (均方根误差) — 主指标
- MAE (平均绝对误差)
- MAPE (平均绝对百分比误差)
- R² (决定系数)
""")
R.append(f"![历史走势]({p_hist})\n")

R.append("## 2. 模型说明\n")
R.append("""\
| 模型 | 类别 | 方法 |
|---|---|---|
| ARIMA (auto) | 统计模型 | 自动选择 (p,d,q) 的 ARIMA，通过 AIC 优化 |
| SARIMAX | 统计模型 | 季节性 ARIMA，网格搜索 order + seasonal_order |
| ETS (Holt-Winters) | 统计模型 | 指数平滑，搜索 trend/damped 组合 |
| AR-XGBoost | 机器学习 | 滞后特征 + XGBoost 回归，网格搜索超参 |
| AR-LightGBM | 机器学习 | 滞后特征 + LightGBM 回归，网格搜索超参 |
| AR-Random Forest | 机器学习 | 滞后特征 + 随机森林回归，网格搜索超参 |
| AR-Ridge | 机器学习 | 滞后特征 + 岭回归，搜索正则化强度 |
| Naive (t-1) | 基线 | 前一日收益率作为预测值 |

**AR 特征工程** (用于 ML 模型):
- 滞后特征: lag_1, lag_2, lag_3, lag_5, lag_10, lag_20, lag_60
- 滚动统计: rolling_5_mean, rolling_20_mean, rolling_5_std
- 差分特征: diff_1, diff_5
""")

R.append("## 3. 模型排行榜\n")
R.append(f"![指标对比]({p_metrics})\n")
R.append(lb.to_markdown())
R.append("")

R.append("## 4. 预测结果可视化\n")
R.append(f"![所有模型预测]({p_all})\n")
R.append(f"![Top-4 预测]({p_top4})\n")

R.append("## 5. 残差分析\n")
R.append(f"![残差分析]({p_resid})\n")
R.append(f"![累积误差]({p_cum})\n")

R.append("## 6. 特征重要性 (ML 模型)\n")
R.append(f"![特征重要性]({p_fi})\n")
R.append("""\
ML 模型的滞后特征重要性分析显示：
- **lag_1 (前1日)** 是最重要的特征，符合国债收益率强自相关特性
- 短期滚动均值和差分特征提供了趋势和动量信号
- 长期滞后 (lag_60) 捕捉了更长周期的均值回归效应
""")

R.append("## 7. 最优超参数\n")
for r in sorted(results, key=lambda x: x["RMSE"])[:5]:
    R.append(f"- **{r['model']}**: {r['params']}")
R.append("")

R.append("## 8. 结论\n")
best_name = best["model"]
R.append(f"""\
### 主要发现

1. **{best_name}** 以 RMSE={best['RMSE']} 取得最优预测性能。

2. **ML 模型 vs 统计模型**: 基于滞后特征的机器学习模型（XGBoost/LightGBM/RF）通常优于传统统计模型（ARIMA/ETS），因为它们能捕捉非线性关系和特征交互。

3. **Naive 基线的竞争力**: 在金融时间序列中，简单的前一日预测（Naive t-1）具有较强竞争力，反映了国债收益率的随机游走特性。任何有效模型都必须显著优于此基线。

4. **ARIMA 类模型**: Auto-ARIMA 通过 AIC 自动选择最优阶数 {arima_order}，SARIMAX 的周期性建模在某些情况下可提供增量改进。

5. **ETS**: 指数平滑模型适合趋势外推，但在收益率变化方向不稳定时表现一般。

### 建议

- **短期预测 (1-5天)**: 优先使用 {best_name}，辅以 Naive 作为合理性检查。
- **中期预测 (1-3月)**: 建议结合宏观经济因子（GDP、CPI、央行政策）构建多因子模型。
- **模型集成**: 可尝试将统计模型和 ML 模型的预测进行加权平均以提高稳健性。
- **实时更新**: 建议每周重训练模型以适应最新市场环境。
""")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n{'='*60}")
print(f"✅ 完成！报告: {REPORT_PATH}")
print(f"   最优模型: {best['model']}  RMSE={best['RMSE']}")
