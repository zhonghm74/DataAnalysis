"""
LSTM-based trading signal model.

Input: past N days of [OHLCV + technical indicators] → LSTM → P(up) probability.
Signal: P(up) > threshold → buy; P(up) < (1-threshold) → sell.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class _TSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        context = (w * out).sum(dim=1)
        return self.head(context).squeeze(-1)


FEATURE_COLS = [
    "open_norm", "high_norm", "low_norm", "close_norm", "volume_norm",
    "rsi14_norm", "macd_hist_norm", "kdj_j_norm", "bb_width",
    "vol_ratio_norm", "pct_chg_norm",
]


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize features for LSTM input."""
    feat = pd.DataFrame(index=df.index)

    close_ma = df["close"].rolling(60).mean()
    close_std = df["close"].rolling(60).std()

    feat["open_norm"] = (df["open"] - close_ma) / (close_std + 1e-10)
    feat["high_norm"] = (df["high"] - close_ma) / (close_std + 1e-10)
    feat["low_norm"] = (df["low"] - close_ma) / (close_std + 1e-10)
    feat["close_norm"] = (df["close"] - close_ma) / (close_std + 1e-10)
    feat["volume_norm"] = df["volume"] / (df["volume"].rolling(20).mean() + 1)
    feat["rsi14_norm"] = (df.get("rsi14", 50) - 50) / 50
    feat["macd_hist_norm"] = df.get("macd_hist", 0) / (df["close"].rolling(20).std() + 1e-10)
    feat["kdj_j_norm"] = (df.get("kdj_j", 50) - 50) / 50
    feat["bb_width"] = df.get("bb_width", 0)
    feat["vol_ratio_norm"] = (df.get("vol_ratio", 1) - 1).clip(-2, 2)
    feat["pct_chg_norm"] = df.get("pct_chg", 0) / 5

    return feat.replace([np.inf, -np.inf], 0).fillna(0)


class LSTMSignalModel:
    """LSTM model that outputs buy/sell signals based on predicted up-probability."""

    def __init__(self, seq_len: int = 30, hidden_size: int = 64,
                 predict_days: int = 3, buy_threshold: float = 0.6,
                 sell_threshold: float = 0.4, epochs: int = 50):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.predict_days = predict_days
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.epochs = epochs
        self.model = None
        self.metrics = {}

    def fit(self, df: pd.DataFrame) -> "LSTMSignalModel":
        """Train LSTM on the stock's historical data."""
        features = _prepare_features(df)
        values = features.values.astype(np.float32)
        n_features = values.shape[1]

        # Labels: 1 if price goes up in predict_days, else 0
        future_ret = df["close"].shift(-self.predict_days) / df["close"] - 1
        labels = (future_ret > 0).astype(float).values

        # Build sequences
        X_list, y_list = [], []
        start = max(60, self.seq_len)  # skip warmup
        for i in range(start, len(values) - self.predict_days):
            X_list.append(values[i - self.seq_len:i])
            y_list.append(labels[i])

        if len(X_list) < 50:
            return self

        X = np.array(X_list)
        y = np.array(y_list)

        # Train/val split (time-based)
        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        train_ds = _TSDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        self.model = _LSTMModel(n_features, self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # Validate
        self.model.eval()
        with torch.no_grad():
            val_proba = self.model(torch.FloatTensor(X_val)).numpy()
            val_pred = (val_proba > 0.5).astype(int)
            from sklearn.metrics import accuracy_score, f1_score
            self.metrics = {
                "训练样本": len(y_tr),
                "验证样本": len(y_val),
                "验证准确率": f"{accuracy_score(y_val, val_pred):.1%}",
                "验证F1": f"{f1_score(y_val, val_pred, zero_division=0):.3f}",
                "预测天数": self.predict_days,
            }
        return self

    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate buy/sell signals for the full DataFrame."""
        signals = pd.Series(0, index=df.index)

        if self.model is None:
            return signals

        features = _prepare_features(df)
        values = features.values.astype(np.float32)

        self.model.eval()
        start = max(60, self.seq_len)
        probas = np.full(len(df), 0.5)

        with torch.no_grad():
            for i in range(start, len(values)):
                seq = values[i - self.seq_len:i]
                x = torch.FloatTensor(seq).unsqueeze(0)
                prob = self.model(x).item()
                probas[i] = prob

        signals[probas > self.buy_threshold] = 1
        signals[probas < self.sell_threshold] = -1

        return signals

    def predict_proba_series(self, df: pd.DataFrame) -> pd.Series:
        """Return raw P(up) probability series for visualization."""
        if self.model is None:
            return pd.Series(0.5, index=df.index)

        features = _prepare_features(df)
        values = features.values.astype(np.float32)
        probas = np.full(len(df), 0.5)
        start = max(60, self.seq_len)

        self.model.eval()
        with torch.no_grad():
            for i in range(start, len(values)):
                seq = values[i - self.seq_len:i]
                x = torch.FloatTensor(seq).unsqueeze(0)
                probas[i] = self.model(x).item()

        return pd.Series(probas, index=df.index)
