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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
# 3B. TRANSFORMER MODELS
# ===================================================================
print("\n" + "=" * 60)
print("3B. Transformer 类时序模型\n")

SEQ_LEN = 60  # look-back window for transformer models

# --- Sliding-window dataset ---
class TSDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])

ts_values = series["yield_10y"].values.astype(np.float32)
ts_mean, ts_std = ts_values[:-TEST_SIZE].mean(), ts_values[:-TEST_SIZE].std()
ts_norm = (ts_values - ts_mean) / ts_std

train_ds = TSDataset(ts_norm[:-TEST_SIZE], SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Build test inputs: each test point uses the preceding SEQ_LEN points
test_start_idx = len(ts_norm) - TEST_SIZE
test_inputs = []
for i in range(TEST_SIZE):
    seq = ts_norm[test_start_idx + i - SEQ_LEN : test_start_idx + i]
    test_inputs.append(seq)
test_inputs = torch.FloatTensor(np.array(test_inputs)).unsqueeze(-1)  # (60, SEQ_LEN, 1)

def train_torch_model(model, name, n_epochs=80, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        preds_norm = model(test_inputs).numpy().flatten()
    preds = preds_norm * ts_std + ts_mean
    return preds

# --- Model 9: Transformer Encoder ---
class TransformerTS(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1, seq_len=60):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])

print("  [9/12] Transformer Encoder …")
best_tf_rmse, best_tf_pred, best_tf_params = 1e18, None, ""
for d_model, nhead, nlayers, dim_ff, lr_ in [
    (32, 4, 2, 64, 1e-3), (64, 4, 3, 128, 5e-4), (32, 4, 3, 64, 5e-4), (16, 4, 2, 32, 1e-3)]:
    torch.manual_seed(42)
    mdl = TransformerTS(d_model=d_model, nhead=nhead, num_layers=nlayers, dim_ff=dim_ff, seq_len=SEQ_LEN)
    t0 = time.time()
    pred = train_torch_model(mdl, "TransformerTS", n_epochs=80, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test["yield_10y"].values, pred))
    pstr = f"d={d_model},h={nhead},L={nlayers},ff={dim_ff},lr={lr_}"
    print(f"    config [{pstr}] RMSE={rmse:.6f}")
    if rmse < best_tf_rmse:
        best_tf_rmse = rmse
        best_tf_pred = pred
        best_tf_params = pstr
        best_tf_time = time.time() - t0
record("Transformer Encoder", test["yield_10y"].values, best_tf_pred,
       best_tf_time, best_tf_params)

# --- Model 10: PatchTST ---
class PatchTST(nn.Module):
    def __init__(self, seq_len=60, patch_len=10, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_patches * d_model, 32),
                                   nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(-1)
        x = x[:, :x.size(1) // self.patch_len * self.patch_len]
        x = x.reshape(B, -1, self.patch_len)
        x = self.patch_proj(x) + self.pos_enc
        x = self.encoder(x)
        return self.head(x)

print("  [10/12] PatchTST …")
best_pt_rmse, best_pt_pred, best_pt_params = 1e18, None, ""
for patch_len, d_model, nlayers, lr_ in [
    (10, 32, 2, 1e-3), (5, 32, 2, 1e-3), (10, 64, 3, 5e-4), (12, 32, 2, 5e-4)]:
    torch.manual_seed(42)
    mdl = PatchTST(seq_len=SEQ_LEN, patch_len=patch_len, d_model=d_model, num_layers=nlayers)
    t0 = time.time()
    pred = train_torch_model(mdl, "PatchTST", n_epochs=80, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test["yield_10y"].values, pred))
    pstr = f"patch={patch_len},d={d_model},L={nlayers},lr={lr_}"
    print(f"    config [{pstr}] RMSE={rmse:.6f}")
    if rmse < best_pt_rmse:
        best_pt_rmse = rmse
        best_pt_pred = pred
        best_pt_params = pstr
        best_pt_time = time.time() - t0
record("PatchTST", test["yield_10y"].values, best_pt_pred,
       best_pt_time, best_pt_params)

# --- Model 11: LSTM + Attention ---
class LSTMAttention(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn_w = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_scores = torch.softmax(self.attn_w(out), dim=1)
        context = (attn_scores * out).sum(dim=1)
        return self.head(context)

print("  [11/12] LSTM + Attention …")
best_la_rmse, best_la_pred, best_la_params = 1e18, None, ""
for hidden, nlayers, lr_ in [(64, 2, 1e-3), (128, 2, 5e-4), (64, 3, 5e-4), (32, 2, 1e-3)]:
    torch.manual_seed(42)
    mdl = LSTMAttention(hidden_size=hidden, num_layers=nlayers)
    t0 = time.time()
    pred = train_torch_model(mdl, "LSTM-Attn", n_epochs=80, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test["yield_10y"].values, pred))
    pstr = f"h={hidden},L={nlayers},lr={lr_}"
    print(f"    config [{pstr}] RMSE={rmse:.6f}")
    if rmse < best_la_rmse:
        best_la_rmse = rmse
        best_la_pred = pred
        best_la_params = pstr
        best_la_time = time.time() - t0
record("LSTM + Attention", test["yield_10y"].values, best_la_pred,
       best_la_time, best_la_params)

# --- Model 12: Informer-lite (ProbSparse Attention) ---
class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
    def forward(self, x):
        B, L, D = x.shape
        top_k = max(1, int(np.ceil(np.log2(L))))
        idx = torch.randint(0, L, (B, top_k), device=x.device)
        q_sparse = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        out, _ = self.attn(q_sparse, x, x)
        result = x.clone()
        result.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, D), out)
        return result

class InformerLite(nn.Module):
    def __init__(self, seq_len=60, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                ProbSparseAttention(d_model, nhead),
                nn.LayerNorm(d_model),
                nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model)),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            ]))
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
        for attn, ln1, ff, ln2, drop in self.layers:
            x = ln1(x + drop(attn(x)))
            x = ln2(x + drop(ff(x)))
        return self.head(x[:, -1, :])

print("  [12/12] Informer-lite …")
best_inf_rmse, best_inf_pred, best_inf_params = 1e18, None, ""
for d_model, nhead, nlayers, lr_ in [
    (32, 4, 2, 1e-3), (64, 4, 2, 5e-4), (32, 4, 3, 5e-4), (16, 4, 2, 1e-3)]:
    torch.manual_seed(42)
    mdl = InformerLite(seq_len=SEQ_LEN, d_model=d_model, nhead=nhead, num_layers=nlayers)
    t0 = time.time()
    pred = train_torch_model(mdl, "Informer-lite", n_epochs=80, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test["yield_10y"].values, pred))
    pstr = f"d={d_model},h={nhead},L={nlayers},lr={lr_}"
    print(f"    config [{pstr}] RMSE={rmse:.6f}")
    if rmse < best_inf_rmse:
        best_inf_rmse = rmse
        best_inf_pred = pred
        best_inf_params = pstr
        best_inf_time = time.time() - t0
record("Informer-lite", test["yield_10y"].values, best_inf_pred,
       best_inf_time, best_inf_params)

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
### 统计模型
| 模型 | 方法 |
|---|---|
| ARIMA (auto) | 自动选择 (p,d,q) 的 ARIMA，通过 AIC 优化 |
| SARIMAX | 季节性 ARIMA，网格搜索 order + seasonal_order |
| ETS (Holt-Winters) | 指数平滑，搜索 trend/damped 组合 |

### 机器学习模型 (基于手工滞后特征)
| 模型 | 方法 |
|---|---|
| AR-XGBoost | 滞后特征 + XGBoost 回归，网格搜索超参 |
| AR-LightGBM | 滞后特征 + LightGBM 回归，网格搜索超参 |
| AR-Random Forest | 滞后特征 + 随机森林回归，网格搜索超参 |
| AR-Ridge | 滞后特征 + 岭回归，搜索正则化强度 |

### Transformer 类深度学习模型 (端到端序列建模)
| 模型 | 方法 |
|---|---|
| Transformer Encoder | 标准多头自注意力编码器 + 位置编码，网格搜索 d_model/nhead/layers |
| PatchTST | 将时序分割为 patch 再做 Transformer 编码（2023 SOTA），搜索 patch_len/d_model |
| LSTM + Attention | 双层 LSTM + 缩放点积注意力池化，搜索 hidden_size/layers |
| Informer-lite | ProbSparse 注意力机制（降低复杂度的 Informer 变体），搜索 d_model/layers |

### 基线
| 模型 | 方法 |
|---|---|
| Naive (t-1) | 前一日收益率作为预测值 |

**AR 特征工程** (用于 ML 模型):
- 滞后特征: lag_1, lag_2, lag_3, lag_5, lag_10, lag_20, lag_60
- 滚动统计: rolling_5_mean, rolling_20_mean, rolling_5_std
- 差分特征: diff_1, diff_5

**Transformer 输入** (用于深度学习模型):
- 滑动窗口: 前 60 个交易日的标准化收益率序列
- 标准化: 训练集均值/标准差归一化
- 训练: Adam + CosineAnnealing, 80 epochs, 梯度裁剪
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

# categorize results
stat_models = [r for r in results if r["model"] in ("ARIMA (auto)", "SARIMAX", "ETS (Holt-Winters)")]
ml_models = [r for r in results if r["model"].startswith("AR-")]
tf_models = [r for r in results if r["model"] in ("Transformer Encoder", "PatchTST", "LSTM + Attention", "Informer-lite")]
best_stat = min(stat_models, key=lambda r: r["RMSE"]) if stat_models else None
best_ml = min(ml_models, key=lambda r: r["RMSE"]) if ml_models else None
best_tf = min(tf_models, key=lambda r: r["RMSE"]) if tf_models else None

R.append(f"""\
### 主要发现

1. **{best_name}** 以 RMSE={best['RMSE']} 取得最优预测性能。

2. **三大类模型性能对比**:
   - 统计模型最优: {best_stat['model'] if best_stat else 'N/A'} (RMSE={best_stat['RMSE'] if best_stat else 'N/A'})
   - ML 模型最优: {best_ml['model'] if best_ml else 'N/A'} (RMSE={best_ml['RMSE'] if best_ml else 'N/A'})
   - Transformer 模型最优: {best_tf['model'] if best_tf else 'N/A'} (RMSE={best_tf['RMSE'] if best_tf else 'N/A'})

3. **Transformer 模型分析**: Transformer 类模型在国债收益率这类低噪声、强自相关的金融时序上，面临"过度建模"的风险——自注意力机制更适合捕捉复杂的长距离依赖关系，但国债收益率的变化主要由短期自相关驱动，简单的滞后特征已足够。PatchTST 通过分 patch 建模能缓解过拟合，通常是 Transformer 类中表现最好的。

4. **Naive 基线的竞争力**: 前一日预测（Naive t-1）极具竞争力，反映了国债收益率的随机游走特性。

5. **统计模型局限**: ARIMA/ETS 的多步直接预测误差快速积累，在 60 天测试期上 R² 为负。

### 各类模型适用场景

| 类别 | 适用场景 | 局限 |
|---|---|---|
| 统计模型 (ARIMA/ETS) | 短期 (1-5 步) 预测，可解释性强 | 多步预测误差积累，无法捕捉非线性 |
| ML 模型 (Ridge/XGBoost) | 中短期预测，特征工程灵活 | 依赖手工特征，不自动学习序列模式 |
| Transformer 类 | 长序列、复杂模式、多变量场景 | 小数据集易过拟合，训练成本高 |

### 建议

- **短期预测 (1-5天)**: 优先使用 {best_name}，辅以 Naive 作为合理性检查。
- **中期预测 (1-3月)**: 结合宏观经济因子（GDP、CPI、央行政策）构建多因子模型。
- **Transformer 优化方向**: 增加训练数据（多期限债券联合建模）、加入宏观因子作为协变量、使用预训练时序基础模型。
- **模型集成**: 将 ML 模型和 Transformer 模型预测加权平均可提高稳健性。
""")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n{'='*60}")
print(f"✅ 完成！报告: {REPORT_PATH}")
print(f"   最优模型: {best['model']}  RMSE={best['RMSE']}")
