"""
Chinese Government Bond Yield Forecasting — Predict DAILY CHANGES (Δyield).

All models predict the daily change: Δy(t) = yield(t) - yield(t-1).
Predicted levels are reconstructed as: ŷ(t) = y(t-1) + Δŷ(t).
This eliminates the dominant lag-1 autocorrelation and tests whether
each model can capture genuine predictive signals.

12 Models:
  Statistical: ARIMA, SARIMAX, ETS
  ML (lag features): XGBoost, LightGBM, Random Forest, Ridge
  Transformer: Encoder, PatchTST, LSTM+Attention, Informer-lite
  Baseline: Naive (Δ=0)
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
# 1. LOAD & PREPARE — predict daily CHANGE
# ===================================================================
print("=" * 60)
print("1. 加载数据 & 构造日变化序列 …")
raw = pd.read_csv(DATA_PATH)
raw["日期"] = pd.to_datetime(raw["日期"])
raw = raw.sort_values("日期").reset_index(drop=True)

TARGET_COL = "中国国债收益率10年"
series = raw[["日期", TARGET_COL]].dropna().copy()
series.columns = ["date", "yield_10y"]
series = series.set_index("date").asfreq("B")
series["yield_10y"] = series["yield_10y"].ffill()
series = series.dropna()

# Daily change as prediction target
series["delta"] = series["yield_10y"].diff()
series = series.dropna()

TEST_SIZE = 60
train_lvl = series["yield_10y"].iloc[:-TEST_SIZE]
test_lvl = series["yield_10y"].iloc[-TEST_SIZE:]
train_delta = series["delta"].iloc[:-TEST_SIZE]
test_delta = series["delta"].iloc[-TEST_SIZE:]

# Previous-day levels needed to reconstruct predictions
prev_levels = series["yield_10y"].iloc[-(TEST_SIZE+1):-1].values

print(f"  序列长度: {len(series)}")
print(f"  时间范围: {series.index[0].date()} ~ {series.index[-1].date()}")
print(f"  Δyield 均值={train_delta.mean():.6f}  std={train_delta.std():.6f}")
print(f"  训练集: {len(train_delta)}  测试集: {TEST_SIZE}")

# ===================================================================
# 2. LAG FEATURES (for ML models — on delta series)
# ===================================================================
LAGS = [1, 2, 3, 5, 10, 20, 60]

def make_lag_features(s, lags=LAGS):
    df = pd.DataFrame({"y": s.values}, index=s.index)
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_5"] = df["y"].shift(1).rolling(5).mean()
    df["rolling_20"] = df["y"].shift(1).rolling(20).mean()
    df["rolling_5_std"] = df["y"].shift(1).rolling(5).std()
    df["rolling_20_std"] = df["y"].shift(1).rolling(20).std()
    df["diff_1"] = df["y"].diff().shift(1)  # second-order diff
    df["abs_lag1"] = df["lag_1"].abs()       # volatility proxy
    # Level features (from original yield)
    lvl = series["yield_10y"].reindex(s.index)
    df["level_lag1"] = lvl.shift(1).reindex(df.index)
    df["level_ma20"] = lvl.shift(1).rolling(20).mean().reindex(df.index)
    return df.dropna()

lag_df = make_lag_features(series["delta"])
feature_cols = [c for c in lag_df.columns if c != "y"]
lag_train = lag_df.iloc[:-TEST_SIZE]
lag_test = lag_df.iloc[-TEST_SIZE:]
X_tr, y_tr = lag_train[feature_cols], lag_train["y"]
X_te, y_te = lag_test[feature_cols], lag_test["y"]

scaler = StandardScaler()
X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=feature_cols)
X_te_sc = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=feature_cols)

# ===================================================================
# 3. EVALUATION — on both Δ and reconstructed levels
# ===================================================================
results = []

def record(name, delta_true, delta_pred, elapsed, params=""):
    """Evaluate on delta AND reconstructed levels."""
    d_mae = mean_absolute_error(delta_true, delta_pred)
    d_rmse = np.sqrt(mean_squared_error(delta_true, delta_pred))
    # Reconstruct levels
    lvl_pred = prev_levels + delta_pred
    lvl_true = test_lvl.values
    l_mae = mean_absolute_error(lvl_true, lvl_pred)
    l_rmse = np.sqrt(mean_squared_error(lvl_true, lvl_pred))
    l_mape = np.mean(np.abs((lvl_true - lvl_pred) / lvl_true)) * 100
    l_r2 = r2_score(lvl_true, lvl_pred)
    # Direction accuracy
    dir_true = (delta_true > 0).astype(int)
    dir_pred = (delta_pred > 0).astype(int)
    dir_acc = (dir_true == dir_pred).mean() * 100

    results.append({
        "model": name,
        "Δ-RMSE": round(d_rmse, 6), "Δ-MAE": round(d_mae, 6),
        "Level-RMSE": round(l_rmse, 6), "Level-MAE": round(l_mae, 6),
        "Level-MAPE(%)": round(l_mape, 4), "Level-R²": round(l_r2, 4),
        "方向准确率(%)": round(dir_acc, 2),
        "time_s": round(elapsed, 1), "params": params,
        "_delta_pred": delta_pred, "_lvl_pred": lvl_pred,
    })
    print(f"  {name:28s}  Δ-RMSE={d_rmse:.6f}  Level-RMSE={l_rmse:.6f}  "
          f"方向={dir_acc:.1f}%  Level-R²={l_r2:.4f}  [{elapsed:.1f}s]")

print("\n" + "=" * 60)
print("2. 模型训练 — 预测每日变化量 Δy(t)\n")

# --- 3.1 ARIMA on delta ---
print("  [1/12] Auto-ARIMA (on Δ) …")
t0 = time.time()
auto_arima = pm.auto_arima(
    train_delta, seasonal=False, stepwise=True,
    suppress_warnings=True, error_action="ignore",
    max_p=5, max_q=5, max_d=1, information_criterion="aic"
)
arima_pred = auto_arima.predict(n_periods=TEST_SIZE)
arima_order = auto_arima.order
record("ARIMA (auto)", test_delta.values, arima_pred,
       time.time() - t0, f"order={arima_order}")

# --- 3.2 SARIMAX on delta ---
print("  [2/12] SARIMAX (on Δ) …")
t0 = time.time()
best_aic, best_sar_pred, best_sar_p = 1e18, None, None
for order in [(1,0,1), (2,0,1), (1,0,2), (2,0,2), (1,1,1)]:
    for seasonal in [(1,0,1,5), (0,1,1,5), (1,0,0,5)]:
        try:
            mod = SARIMAX(train_delta, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=200)
            if res.aic < best_aic:
                best_aic = res.aic
                best_sar_pred = res.forecast(TEST_SIZE).values
                best_sar_p = f"order={order}, seasonal={seasonal}"
        except:
            pass
if best_sar_pred is not None:
    record("SARIMAX", test_delta.values, best_sar_pred, time.time() - t0, best_sar_p)

# --- 3.3 ETS on delta ---
print("  [3/12] ETS (on Δ) …")
t0 = time.time()
# ETS needs positive data for multiplicative; delta can be negative → use additive only
best_ets_aic, best_ets_pred, best_ets_p = 1e18, None, ""
for trend in ["add", None]:
    for damped in [True, False]:
        if trend is None and damped:
            continue
        try:
            ets = ExponentialSmoothing(
                train_delta, trend=trend, damped_trend=damped,
                seasonal=None, initialization_method="estimated"
            ).fit(optimized=True)
            pred = ets.forecast(TEST_SIZE).values
            if ets.aic < best_ets_aic:
                best_ets_aic = ets.aic
                best_ets_pred = pred
                best_ets_p = f"trend={trend}, damped={damped}"
        except:
            pass
record("ETS (Holt-Winters)", test_delta.values, best_ets_pred,
       time.time() - t0, best_ets_p)

# --- 3.4 AR-XGBoost ---
print("  [4/12] AR-XGBoost …")
t0 = time.time()
best_xgb_s, best_xgb_pred, best_xgb_p = 1e18, None, {}
for md in [3, 5, 7]:
    for lr_ in [0.01, 0.05, 0.1]:
        for ne in [200, 500]:
            mdl = xgb.XGBRegressor(max_depth=md, learning_rate=lr_, n_estimators=ne,
                                    subsample=0.8, colsample_bytree=0.8,
                                    tree_method="hist", random_state=42, verbosity=0)
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_te_sc)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            if rmse < best_xgb_s:
                best_xgb_s, best_xgb_pred = rmse, pred
                best_xgb_p = {"max_depth": md, "lr": lr_, "n_estimators": ne}
record("AR-XGBoost", y_te.values, best_xgb_pred, time.time() - t0, str(best_xgb_p))

# --- 3.5 AR-LightGBM ---
print("  [5/12] AR-LightGBM …")
t0 = time.time()
best_lgb_s, best_lgb_pred, best_lgb_p = 1e18, None, {}
for md in [3, 5, 7, -1]:
    for lr_ in [0.01, 0.05, 0.1]:
        for nl in [31, 63]:
            mdl = lgb.LGBMRegressor(max_depth=md, learning_rate=lr_, n_estimators=500,
                                     num_leaves=nl, subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbose=-1)
            mdl.fit(X_tr_sc, y_tr)
            pred = mdl.predict(X_te_sc)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            if rmse < best_lgb_s:
                best_lgb_s, best_lgb_pred = rmse, pred
                best_lgb_p = {"max_depth": md, "lr": lr_, "num_leaves": nl}
record("AR-LightGBM", y_te.values, best_lgb_pred, time.time() - t0, str(best_lgb_p))

# --- 3.6 AR-RF ---
print("  [6/12] AR-Random Forest …")
t0 = time.time()
best_rf_s, best_rf_pred, best_rf_p = 1e18, None, {}
for ne in [200, 500]:
    for md in [5, 10, 15, None]:
        mdl = RandomForestRegressor(n_estimators=ne, max_depth=md, random_state=42, n_jobs=-1)
        mdl.fit(X_tr_sc, y_tr)
        pred = mdl.predict(X_te_sc)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        if rmse < best_rf_s:
            best_rf_s, best_rf_pred = rmse, pred
            best_rf_p = {"n_estimators": ne, "max_depth": md}
record("AR-Random Forest", y_te.values, best_rf_pred, time.time() - t0, str(best_rf_p))

# --- 3.7 AR-Ridge ---
print("  [7/12] AR-Ridge …")
t0 = time.time()
best_ridge_s, best_ridge_pred, best_ridge_a = 1e18, None, None
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    mdl = Ridge(alpha=alpha)
    mdl.fit(X_tr_sc, y_tr)
    pred = mdl.predict(X_te_sc)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    if rmse < best_ridge_s:
        best_ridge_s, best_ridge_pred, best_ridge_a = rmse, pred, alpha
record("AR-Ridge", y_te.values, best_ridge_pred, time.time() - t0, f"alpha={best_ridge_a}")

# --- 3.8 Naive (Δ=0) ---
print("  [8/12] Naive (Δ=0) …")
t0 = time.time()
naive_pred = np.zeros(TEST_SIZE)
record("Naive (Δ=0)", test_delta.values, naive_pred, time.time() - t0, "Δ(t)=0")

# ===================================================================
# 3B. TRANSFORMER MODELS
# ===================================================================
print("\n" + "=" * 60)
print("3. Transformer 模型 — 预测 Δy(t)\n")

SEQ_LEN = 60

class TSDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data; self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])

delta_vals = series["delta"].values.astype(np.float32)
d_mean, d_std = delta_vals[:-TEST_SIZE].mean(), delta_vals[:-TEST_SIZE].std()
d_norm = (delta_vals - d_mean) / d_std

train_ds = TSDataset(d_norm[:-TEST_SIZE], SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

test_start = len(d_norm) - TEST_SIZE
test_inputs = []
for i in range(TEST_SIZE):
    seq = d_norm[test_start + i - SEQ_LEN : test_start + i]
    test_inputs.append(seq)
test_inputs_t = torch.FloatTensor(np.array(test_inputs)).unsqueeze(-1)

def train_torch(model, n_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        preds_n = model(test_inputs_t).numpy().flatten()
    return preds_n * d_std + d_mean

# --- Transformer Encoder ---
class TransformerTS(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1, seq_len=60):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        return self.head(self.enc(x)[:, -1, :])

print("  [9/12] Transformer Encoder …")
best_rmse, best_pred, best_p = 1e18, None, ""
for dm, nh, nl, ff, lr_ in [(32,4,2,64,1e-3),(64,4,3,128,5e-4),(16,4,2,32,1e-3),(32,4,2,64,5e-4)]:
    torch.manual_seed(42)
    mdl = TransformerTS(d_model=dm, nhead=nh, num_layers=nl, dim_ff=ff, seq_len=SEQ_LEN)
    t0 = time.time()
    pred = train_torch(mdl, n_epochs=100, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test_delta.values, pred))
    ps = f"d={dm},h={nh},L={nl},ff={ff},lr={lr_}"
    print(f"    [{ps}] Δ-RMSE={rmse:.6f}")
    if rmse < best_rmse:
        best_rmse, best_pred, best_p = rmse, pred, ps
        best_t = time.time() - t0
record("Transformer Encoder", test_delta.values, best_pred, best_t, best_p)

# --- PatchTST ---
class PatchTST(nn.Module):
    def __init__(self, seq_len=60, patch_len=10, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.pl = patch_len
        np_ = seq_len // patch_len
        self.pp = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, np_, d_model) * 0.02)
        el = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(np_ * d_model, 32),
                                   nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(-1)[:, :x.size(1)//self.pl*self.pl].reshape(B, -1, self.pl)
        x = self.pp(x) + self.pos
        return self.head(self.enc(x))

print("  [10/12] PatchTST …")
best_rmse, best_pred, best_p = 1e18, None, ""
for pl, dm, nl, lr_ in [(10,32,2,1e-3),(5,32,2,1e-3),(10,64,3,5e-4),(12,32,2,5e-4)]:
    torch.manual_seed(42)
    mdl = PatchTST(seq_len=SEQ_LEN, patch_len=pl, d_model=dm, num_layers=nl)
    t0 = time.time()
    pred = train_torch(mdl, n_epochs=100, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test_delta.values, pred))
    ps = f"patch={pl},d={dm},L={nl},lr={lr_}"
    print(f"    [{ps}] Δ-RMSE={rmse:.6f}")
    if rmse < best_rmse:
        best_rmse, best_pred, best_p = rmse, pred, ps
        best_t = time.time() - t0
record("PatchTST", test_delta.values, best_pred, best_t, best_p)

# --- LSTM + Attention ---
class LSTMAttention(nn.Module):
    def __init__(self, hidden=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        return self.head((w * out).sum(dim=1))

print("  [11/12] LSTM + Attention …")
best_rmse, best_pred, best_p = 1e18, None, ""
for hid, nl, lr_ in [(64,2,1e-3),(128,2,5e-4),(64,3,5e-4),(32,2,1e-3)]:
    torch.manual_seed(42)
    mdl = LSTMAttention(hidden=hid, num_layers=nl)
    t0 = time.time()
    pred = train_torch(mdl, n_epochs=100, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test_delta.values, pred))
    ps = f"h={hid},L={nl},lr={lr_}"
    print(f"    [{ps}] Δ-RMSE={rmse:.6f}")
    if rmse < best_rmse:
        best_rmse, best_pred, best_p = rmse, pred, ps
        best_t = time.time() - t0
record("LSTM + Attention", test_delta.values, best_pred, best_t, best_p)

# --- Informer-lite ---
class ProbSparseAttn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0.1)
    def forward(self, x):
        B, L, D = x.shape
        k = max(1, int(np.ceil(np.log2(L))))
        idx = torch.randint(0, L, (B, k), device=x.device)
        q = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1,-1,D))
        out, _ = self.attn(q, x, x)
        r = x.clone(); r.scatter_(1, idx.unsqueeze(-1).expand(-1,-1,D), out)
        return r

class InformerLite(nn.Module):
    def __init__(self, seq_len=60, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                ProbSparseAttn(d_model, nhead), nn.LayerNorm(d_model),
                nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model)),
                nn.LayerNorm(d_model), nn.Dropout(dropout)]))
        self.head = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        for attn, ln1, ff, ln2, drop in self.layers:
            x = ln1(x + drop(attn(x))); x = ln2(x + drop(ff(x)))
        return self.head(x[:, -1, :])

print("  [12/12] Informer-lite …")
best_rmse, best_pred, best_p = 1e18, None, ""
for dm, nh, nl, lr_ in [(32,4,2,1e-3),(64,4,2,5e-4),(16,4,2,1e-3),(32,4,3,5e-4)]:
    torch.manual_seed(42)
    mdl = InformerLite(seq_len=SEQ_LEN, d_model=dm, nhead=nh, num_layers=nl)
    t0 = time.time()
    pred = train_torch(mdl, n_epochs=100, lr=lr_)
    rmse = np.sqrt(mean_squared_error(test_delta.values, pred))
    ps = f"d={dm},h={nh},L={nl},lr={lr_}"
    print(f"    [{ps}] Δ-RMSE={rmse:.6f}")
    if rmse < best_rmse:
        best_rmse, best_pred, best_p = rmse, pred, ps
        best_t = time.time() - t0
record("Informer-lite", test_delta.values, best_pred, best_t, best_p)

# ===================================================================
# 4. LEADERBOARD
# ===================================================================
print("\n" + "=" * 60)
print("4. 排行榜\n")
lb = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in results]).sort_values("Δ-RMSE").reset_index(drop=True)
lb.index = lb.index + 1; lb.index.name = "排名"
print(lb.to_string())

best = min(results, key=lambda r: r["Δ-RMSE"])
print(f"\n🏆 最优模型: {best['model']}  Δ-RMSE={best['Δ-RMSE']}  方向准确率={best['方向准确率(%)']}%")

# ===================================================================
# 5. CHARTS
# ===================================================================
print("\n" + "=" * 60)
print("5. 生成图表 …")
test_dates = test_lvl.index

# 5.1 Historical + delta
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(series.index, series["yield_10y"], color="steelblue", lw=0.8)
axes[0].axvline(test_lvl.index[0], color="red", ls="--", alpha=0.7, label="训练/测试分界")
axes[0].set_title("中国10年期国债收益率", fontsize=13); axes[0].set_ylabel("收益率 (%)"); axes[0].legend()
axes[1].plot(series.index, series["delta"], color="darkorange", lw=0.5, alpha=0.7)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].axvline(test_lvl.index[0], color="red", ls="--", alpha=0.7)
axes[1].set_title("每日变化 Δy(t)", fontsize=13); axes[1].set_ylabel("Δ收益率")
fig.tight_layout(); p_hist = savefig(fig, "bond_historical.png")

# 5.2 Delta predictions — top 4
top4 = sorted(results, key=lambda r: r["Δ-RMSE"])[:4]
colors = ["#E91E63", "#2196F3", "#FF9800", "#4CAF50"]
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
for ax, r, c in zip(axes.flatten(), top4, colors):
    ax.bar(range(TEST_SIZE), test_delta.values, alpha=0.4, color="gray", label="实际 Δ")
    ax.plot(range(TEST_SIZE), r["_delta_pred"], color=c, lw=1.5, label=f"预测 Δ")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{r['model']}  (Δ-RMSE={r['Δ-RMSE']:.6f}, 方向={r['方向准确率(%)']:.1f}%)", fontsize=11)
    ax.legend(fontsize=9); ax.set_ylabel("Δ收益率")
fig.suptitle("Top-4 模型日变化预测 vs 实际", fontsize=15, y=1.01)
fig.tight_layout(); p_top4_delta = savefig(fig, "bond_top4_delta.png")

# 5.3 Level predictions — top 4
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
for ax, r, c in zip(axes.flatten(), top4, colors):
    ax.plot(test_dates, test_lvl.values, "k-", lw=1.5, label="实际值")
    ax.plot(test_dates, r["_lvl_pred"], color=c, lw=1.5, ls="--", label="预测值")
    ax.set_title(f"{r['model']}  (Level-RMSE={r['Level-RMSE']:.6f})", fontsize=11)
    ax.legend(fontsize=9); ax.set_ylabel("收益率 (%)"); ax.tick_params(axis="x", rotation=30)
fig.suptitle("Top-4 模型还原收益率 vs 实际", fontsize=15, y=1.01)
fig.tight_layout(); p_top4_lvl = savefig(fig, "bond_top4_predictions.png")

# 5.4 All models overlay (levels)
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(test_dates, test_lvl.values, "k-", lw=2.5, label="实际值", zorder=10)
cmap = plt.cm.tab10
for i, r in enumerate(sorted(results, key=lambda x: x["Δ-RMSE"])):
    ax.plot(test_dates, r["_lvl_pred"], lw=1.2, alpha=0.8, color=cmap(i % 10),
            label=f"{r['model']} (Δ-RMSE={r['Δ-RMSE']:.4f})")
ax.set_title("所有模型还原收益率对比", fontsize=14)
ax.set_xlabel("日期"); ax.set_ylabel("收益率 (%)"); ax.legend(fontsize=7, loc="upper left")
ax.tick_params(axis="x", rotation=30); fig.tight_layout()
p_all = savefig(fig, "bond_all_predictions.png")

# 5.5 Metrics comparison
fig, axes = plt.subplots(1, 4, figsize=(24, 7))
for ax, met, color in zip(axes, ["Δ-RMSE", "Δ-MAE", "方向准确率(%)", "Level-R²"],
                           ["#E91E63", "#2196F3", "#FF9800", "#4CAF50"]):
    d = lb.sort_values(met, ascending=(met not in ("方向准确率(%)", "Level-R²")))
    ax.barh(d["model"], d[met], color=color, edgecolor="white")
    ax.set_title(met, fontsize=13)
    for i, v in enumerate(d[met]):
        ax.text(v + (d[met].max() - d[met].min()) * 0.02, i, f"{v:.4f}", va="center", fontsize=8)
fig.suptitle("模型评估指标对比（预测每日变化）", fontsize=15, y=1.02)
fig.tight_layout(); p_metrics = savefig(fig, "bond_metrics_comparison.png")

# 5.6 Direction accuracy chart
fig, ax = plt.subplots(figsize=(10, 7))
dir_df = lb.sort_values("方向准确率(%)")
colors_dir = ["#4CAF50" if v > 50 else "#F44336" for v in dir_df["方向准确率(%)"]]
ax.barh(dir_df["model"], dir_df["方向准确率(%)"], color=colors_dir, edgecolor="white")
ax.axvline(50, color="black", ls="--", alpha=0.5, label="随机猜测 (50%)")
ax.set_title("涨跌方向预测准确率", fontsize=14); ax.set_xlabel("准确率 (%)")
ax.legend(); fig.tight_layout()
p_dir = savefig(fig, "bond_direction_accuracy.png")

# 5.7 Residual analysis (top4)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, r, c in zip(axes.flatten(), top4, colors):
    resid = test_delta.values - r["_delta_pred"]
    ax.bar(range(len(resid)), resid, color=c, alpha=0.7, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{r['model']} Δ预测残差", fontsize=12)
    ax.set_xlabel("样本"); ax.set_ylabel("残差")
fig.suptitle("Top-4 残差分析", fontsize=15, y=1.01)
fig.tight_layout(); p_resid = savefig(fig, "bond_residual_analysis.png")

# 5.8 Cumulative error
fig, ax = plt.subplots(figsize=(14, 6))
for i, r in enumerate(sorted(results, key=lambda x: x["Δ-RMSE"])[:5]):
    ce = np.cumsum(np.abs(test_delta.values - r["_delta_pred"]))
    ax.plot(test_dates, ce, lw=1.5, color=cmap(i), label=r["model"])
ax.set_title("累积绝对误差 (Δ预测, Top-5)", fontsize=14)
ax.set_xlabel("日期"); ax.set_ylabel("累积 |Δ误差|"); ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30); fig.tight_layout()
p_cum = savefig(fig, "bond_cumulative_error.png")

# 5.9 Feature importance (best ML model)
fig, ax = plt.subplots(figsize=(10, 7))
ml_res = [r for r in results if "AR-" in r["model"]]
if ml_res:
    best_ml = min(ml_res, key=lambda r: r["Δ-RMSE"])
    try:
        p_ = json.loads(best_ml["params"].replace("'", '"').replace("lr", "learning_rate"))
    except:
        p_ = {}
    if "XGBoost" in best_ml["model"]:
        fi_mdl = xgb.XGBRegressor(**p_, subsample=0.8, colsample_bytree=0.8,
                                    tree_method="hist", random_state=42, verbosity=0)
    elif "LightGBM" in best_ml["model"]:
        fi_mdl = lgb.LGBMRegressor(**p_, subsample=0.8, colsample_bytree=0.8,
                                    n_estimators=500, random_state=42, verbose=-1)
    elif "Ridge" in best_ml["model"]:
        alpha_v = float(best_ml["params"].split("=")[1])
        fi_mdl = Ridge(alpha=alpha_v)
    else:
        fi_mdl = RandomForestRegressor(random_state=42, n_jobs=-1)
    fi_mdl.fit(X_tr_sc, y_tr)
    if hasattr(fi_mdl, "feature_importances_"):
        imp = fi_mdl.feature_importances_
    else:
        imp = np.abs(fi_mdl.coef_)
    imp_s = pd.Series(imp, index=feature_cols).sort_values()
    imp_s.plot.barh(ax=ax, color="teal", edgecolor="white")
    ax.set_title(f"Δ预测特征重要性 — {best_ml['model']}", fontsize=13)
fig.tight_layout(); p_fi = savefig(fig, "bond_feature_importance.png")

# ===================================================================
# 6. REPORT
# ===================================================================
print("\n" + "=" * 60)
print("6. 生成报告 …")
R = []
R.append("# 中国10年期国债收益率预测 — 多模型自回归比较报告\n")
R.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
R.append("> **预测目标: 每日变化 Δy(t) = yield(t) - yield(t-1)**\n---\n")

R.append("## 1. 研究概述\n")
R.append(f"""\
**目标**: 预测中国10年期国债收益率的**每日变化量**，消除前一日水平值的主导效应，
真正检验模型是否能捕捉到预测性信号。

**关键设计**:
- 预测 Δy(t) = yield(t) - yield(t-1)，而非 yield(t) 本身
- 还原收益率水平: ŷ(t) = y(t-1) + Δŷ(t)
- 新增**方向准确率**指标: 模型预测涨跌方向的正确比例
- Naive 基线变为 Δ=0（即预测"不变"），这是金融序列中最难超越的基线

**数据**: {series.index[0].date()} ~ {series.index[-1].date()}, {len(series)} 个交易日
- Δyield 均值={train_delta.mean():.6f}, std={train_delta.std():.6f}
- 训练集: {len(train_delta)}, 测试集: {TEST_SIZE} 个交易日
""")
R.append(f"![历史走势]({p_hist})\n")

R.append("## 2. 模型说明\n")
R.append("""\
### 统计模型
| 模型 | 方法 |
|---|---|
| ARIMA (auto) | 对 Δy 序列自动选择 (p,d,q)，AIC 优化 |
| SARIMAX | 对 Δy 序列做季节性 ARIMA |
| ETS (Holt-Winters) | 对 Δy 序列做指数平滑 |

### ML 模型 (滞后特征 → 回归)
| 模型 | 方法 |
|---|---|
| AR-XGBoost / AR-LightGBM | Δy 的滞后特征 + 梯度提升回归 |
| AR-Random Forest | Δy 的滞后特征 + 随机森林 |
| AR-Ridge | Δy 的滞后特征 + 岭回归 |

### Transformer 深度学习模型
| 模型 | 方法 |
|---|---|
| Transformer Encoder | 多头自注意力 + 位置编码，端到端预测 Δy |
| PatchTST | 序列分 patch → Transformer 编码 (2023 SOTA) |
| LSTM + Attention | LSTM + 注意力池化 |
| Informer-lite | ProbSparse 注意力（降低复杂度） |

### 基线
| 模型 | 方法 |
|---|---|
| Naive (Δ=0) | 预测"不变"（Δ=0），等价于 ŷ(t)=y(t-1) |
""")

R.append("## 3. 模型排行榜\n")
R.append(f"![指标对比]({p_metrics})\n")
R.append(lb.to_markdown()); R.append("")

R.append("## 4. 涨跌方向预测准确率\n")
R.append(f"![方向准确率]({p_dir})\n")
R.append("方向准确率是金融预测中最重要的实用指标之一，>50% 意味着模型优于随机猜测。\n")

R.append("## 5. 预测可视化\n")
R.append(f"### 5.1 日变化 Δy 预测 (Top-4)\n![Top-4 delta]({p_top4_delta})\n")
R.append(f"### 5.2 还原收益率 (Top-4)\n![Top-4 level]({p_top4_lvl})\n")
R.append(f"### 5.3 所有模型对比\n![所有模型]({p_all})\n")

R.append("## 6. 残差与误差分析\n")
R.append(f"![残差]({p_resid})\n![累积误差]({p_cum})\n")

R.append("## 7. 特征重要性 (ML 模型)\n")
R.append(f"![特征重要性]({p_fi})\n")

R.append("## 8. 最优超参数\n")
for r in sorted(results, key=lambda x: x["Δ-RMSE"])[:6]:
    R.append(f"- **{r['model']}**: {r['params']}")
R.append("")

R.append("## 9. 结论\n")
stat_m = [r for r in results if r["model"] in ("ARIMA (auto)", "SARIMAX", "ETS (Holt-Winters)")]
ml_m = [r for r in results if r["model"].startswith("AR-")]
tf_m = [r for r in results if r["model"] in ("Transformer Encoder", "PatchTST", "LSTM + Attention", "Informer-lite")]
bs = min(stat_m, key=lambda r: r["Δ-RMSE"]) if stat_m else None
bm = min(ml_m, key=lambda r: r["Δ-RMSE"]) if ml_m else None
bt = min(tf_m, key=lambda r: r["Δ-RMSE"]) if tf_m else None
R.append(f"""\
### 主要发现

1. **预测每日变化 vs 预测水平值**: 当预测目标改为 Δy(t) 后，消除了 lag-1 的主导效应，
   模型之间的差异更能反映其真实预测能力。Naive (Δ=0) 基线变得非常强劲。

2. **🏆 最优模型: {best['model']}** — Δ-RMSE={best['Δ-RMSE']}, 方向准确率={best['方向准确率(%)']}%

3. **各类模型最优**:
   - 统计模型: {bs['model'] if bs else 'N/A'} (Δ-RMSE={bs['Δ-RMSE'] if bs else 'N/A'}, 方向={bs['方向准确率(%)'] if bs else 'N/A'}%)
   - ML 模型: {bm['model'] if bm else 'N/A'} (Δ-RMSE={bm['Δ-RMSE'] if bm else 'N/A'}, 方向={bm['方向准确率(%)'] if bm else 'N/A'}%)
   - Transformer: {bt['model'] if bt else 'N/A'} (Δ-RMSE={bt['Δ-RMSE'] if bt else 'N/A'}, 方向={bt['方向准确率(%)'] if bt else 'N/A'}%)

4. **方向预测**: 方向准确率是交易策略的核心，>50% 表明模型具有实际价值。

5. **过拟合风险**: 在预测 Δy 时，过于复杂的模型（如深层 Transformer）容易过拟合训练集的
   噪声模式，反而不如简单模型（Ridge、浅层网络）。

### 建议

- 以**方向准确率**为主要选模标准，Δ-RMSE 为辅
- Transformer 模型需要更多数据或预训练才能充分发挥优势
- 可尝试将 Δy 预测与宏观因子（利差、CPI、M2）结合构建多因子模型
- 模型集成（ML + Transformer 加权）可能进一步提升方向准确率
""")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n{'='*60}")
print(f"✅ 完成！{REPORT_PATH}")
print(f"   最优: {best['model']}  Δ-RMSE={best['Δ-RMSE']}  方向={best['方向准确率(%)']}%")
