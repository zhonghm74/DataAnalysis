"""
Meta-Labeling: ML filter on top of rule-based signals.

Idea: rule strategies generate raw buy/sell signals, then a LightGBM classifier
decides whether each signal is likely to be profitable ("good" or "bad").
Only signals with high ML confidence are executed.

Reference: Marcos Lopez de Prado, "Advances in Financial Machine Learning"
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _build_meta_features(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """Build features for the meta-labeling classifier around signal points."""
    feat = pd.DataFrame(index=df.index)
    feat["signal"] = signal

    feat["rsi14"] = df.get("rsi14", 50)
    feat["macd_hist"] = df.get("macd_hist", 0)
    feat["bb_width"] = df.get("bb_width", 0)
    feat["kdj_j"] = df.get("kdj_j", 50)
    feat["vol_ratio"] = df.get("vol_ratio", 1)
    feat["atr14"] = df.get("atr14", 0)
    feat["turnover"] = df.get("turnover", 0)
    feat["pct_chg"] = df.get("pct_chg", 0)

    # Trend context
    feat["above_ma5"] = (df["close"] > df.get("ma5", df["close"])).astype(int)
    feat["above_ma20"] = (df["close"] > df.get("ma20", df["close"])).astype(int)
    feat["above_ma60"] = (df["close"] > df.get("ma60", df["close"])).astype(int)
    feat["ma5_slope"] = df.get("ma5", df["close"]).pct_change(5)
    feat["ma20_slope"] = df.get("ma20", df["close"]).pct_change(5)

    # Momentum
    feat["ret_1d"] = df["close"].pct_change(1)
    feat["ret_5d"] = df["close"].pct_change(5)
    feat["ret_20d"] = df["close"].pct_change(20)

    # Volatility
    feat["vol_5d"] = df["close"].pct_change().rolling(5).std()
    feat["vol_20d"] = df["close"].pct_change().rolling(20).std()

    return feat


def _label_signals(df: pd.DataFrame, signal: pd.Series, hold_days: int = 5,
                   profit_threshold: float = 0.0) -> pd.Series:
    """
    Label each signal: 1 = profitable (good signal), 0 = unprofitable (bad signal).
    Buy signals are profitable if price goes up within hold_days.
    Sell signals are profitable if price goes down within hold_days.
    """
    labels = pd.Series(np.nan, index=df.index)
    future_ret = df["close"].shift(-hold_days) / df["close"] - 1

    buy_mask = signal > 0
    sell_mask = signal < 0
    labels[buy_mask] = (future_ret[buy_mask] > profit_threshold).astype(int)
    labels[sell_mask] = (future_ret[sell_mask] < -profit_threshold).astype(int)

    return labels


class MetaLabeler:
    """Train a meta-labeling classifier and filter signals."""

    def __init__(self, hold_days: int = 5, confidence_threshold: float = 0.55):
        self.hold_days = hold_days
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.feature_cols = None
        self.metrics = {}

    def fit(self, df: pd.DataFrame, raw_signal: pd.Series) -> "MetaLabeler":
        features = _build_meta_features(df, raw_signal)
        labels = _label_signals(df, raw_signal, self.hold_days)

        mask = labels.notna() & (raw_signal != 0)
        X = features.loc[mask].drop(columns=["signal"], errors="ignore")
        y = labels.loc[mask].astype(int)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.feature_cols = X.columns.tolist()

        if len(y) < 20:
            self.model = None
            return self

        # Time-series split (no look-ahead)
        tscv = TimeSeriesSplit(n_splits=3)
        best_model, best_score = None, -1

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            mdl = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                num_leaves=15, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
            mdl.fit(X_tr, y_tr)
            score = f1_score(y_val, mdl.predict(X_val), zero_division=0)
            if score > best_score:
                best_score = score
                best_model = mdl

        # Retrain on all data
        self.model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        self.model.fit(X, y)

        y_pred = self.model.predict(X)
        self.metrics = {
            "训练样本": len(y),
            "正样本比例": f"{y.mean():.1%}",
            "准确率": f"{accuracy_score(y, y_pred):.1%}",
            "精确率": f"{precision_score(y, y_pred, zero_division=0):.1%}",
            "召回率": f"{recall_score(y, y_pred, zero_division=0):.1%}",
            "F1": f"{f1_score(y, y_pred, zero_division=0):.3f}",
            "CV最优F1": f"{best_score:.3f}",
        }
        return self

    def filter_signals(self, df: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """Return filtered signal: only keep signals with high ML confidence."""
        if self.model is None:
            return raw_signal

        features = _build_meta_features(df, raw_signal)
        signal_mask = raw_signal != 0
        if signal_mask.sum() == 0:
            return raw_signal

        X = features.loc[signal_mask][self.feature_cols] if self.feature_cols else features.loc[signal_mask]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        proba = self.model.predict_proba(X)[:, 1]
        confident = proba >= self.confidence_threshold

        filtered = raw_signal.copy()
        reject_idx = signal_mask[signal_mask].index[~confident]
        filtered.loc[reject_idx] = 0

        return filtered

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp
