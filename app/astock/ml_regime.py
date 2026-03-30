"""
Market regime detector using Hidden Markov Model (HMM) + feature clustering.

Identifies 3 market states: Bull (牛市), Bear (熊市), Sideways (震荡).
Used to filter trading signals — e.g., only execute buy signals in bull regime.

Uses a simple Gaussian Mixture approach (no hmmlearn dependency) with
rolling features to classify market states.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class MarketRegimeDetector:
    """Detect market regimes (Bull/Bear/Sideways) from price data."""

    REGIME_NAMES = {0: "震荡", 1: "牛市", 2: "熊市"}
    REGIME_COLORS = {"牛市": "#4CAF50", "熊市": "#F44336", "震荡": "#FF9800"}

    def __init__(self, n_regimes: int = 3, lookback: int = 20):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self.regime_map = {}

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime features from price data."""
        feat = pd.DataFrame(index=df.index)
        ret = df["close"].pct_change()

        feat["ret_mean"] = ret.rolling(self.lookback).mean()
        feat["ret_std"] = ret.rolling(self.lookback).std()
        feat["ret_skew"] = ret.rolling(self.lookback).skew()

        feat["trend"] = (df["close"] / df["close"].shift(self.lookback) - 1)

        if "ma5" in df and "ma20" in df:
            feat["ma_spread"] = (df["ma5"] - df["ma20"]) / (df["ma20"] + 1e-10)
        else:
            ma5 = df["close"].rolling(5).mean()
            ma20 = df["close"].rolling(20).mean()
            feat["ma_spread"] = (ma5 - ma20) / (ma20 + 1e-10)

        if "rsi14" in df:
            feat["rsi"] = df["rsi14"] / 100
        else:
            feat["rsi"] = 0.5

        feat["vol_regime"] = ret.rolling(self.lookback).std() / ret.rolling(60).std()

        if "volume" in df:
            feat["vol_trend"] = df["volume"].rolling(5).mean() / (df["volume"].rolling(20).mean() + 1)

        return feat.replace([np.inf, -np.inf], np.nan).dropna()

    def fit(self, df: pd.DataFrame) -> "MarketRegimeDetector":
        """Fit GMM on historical regime features."""
        features = self._extract_features(df)
        if len(features) < 50:
            return self

        X = self.scaler.fit_transform(features)

        self.model = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full",
            n_init=10, random_state=42, max_iter=200
        )
        self.model.fit(X)

        labels = self.model.predict(X)
        avg_ret = features.groupby(labels)["trend"].mean()

        # Map: highest avg return → Bull, lowest → Bear, middle → Sideways
        sorted_labels = avg_ret.sort_values().index.tolist()
        self.regime_map = {
            sorted_labels[-1]: "牛市",
            sorted_labels[0]: "熊市",
        }
        for lbl in sorted_labels:
            if lbl not in self.regime_map:
                self.regime_map[lbl] = "震荡"

        return self

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return regime labels aligned to df index."""
        features = self._extract_features(df)
        regimes = pd.Series("未知", index=df.index)

        if self.model is None or len(features) == 0:
            return regimes

        X = self.scaler.transform(features)
        labels = self.model.predict(X)
        regime_names = [self.regime_map.get(l, "震荡") for l in labels]
        regimes.loc[features.index] = regime_names

        return regimes

    def detect_current(self, df: pd.DataFrame) -> dict:
        """Detect current regime and return summary."""
        regimes = self.detect(df)
        current = regimes.iloc[-1]

        last_30 = regimes.tail(30)
        regime_counts = last_30.value_counts()

        return {
            "当前状态": current,
            "颜色": self.REGIME_COLORS.get(current, "#9E9E9E"),
            "近30日分布": regime_counts.to_dict(),
            "建议": self._get_advice(current),
        }

    @staticmethod
    def _get_advice(regime: str) -> str:
        advice = {
            "牛市": "趋势向上，可积极做多，跟踪趋势策略有效",
            "熊市": "趋势向下，应减仓或空仓，反弹卖出为主",
            "震荡": "方向不明，宜降低仓位，高抛低吸为主",
        }
        return advice.get(regime, "等待更多数据")

    def filter_signals_by_regime(self, signals: pd.Series,
                                  regimes: pd.Series) -> pd.Series:
        """Filter signals based on regime: only buy in bull, only sell in bear."""
        filtered = signals.copy()
        # Suppress buy signals in bear market
        bear_mask = regimes == "熊市"
        filtered[bear_mask & (signals > 0)] = 0
        # Suppress sell signals in bull market (optional, more aggressive)
        # bull_mask = regimes == "牛市"
        # filtered[bull_mask & (signals < 0)] = 0
        return filtered
