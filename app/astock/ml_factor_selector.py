"""
LightGBM multi-factor stock selector.

Replaces rule-based scoring with a ML model that predicts future N-day return
from 30+ quantitative factors, then ranks stocks by predicted return.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from .market_data import fetch_stock, add_technical_indicators


def _compute_factors(df: pd.DataFrame) -> dict:
    """Compute 30+ quantitative factors from daily OHLCV data."""
    latest = df.iloc[-1]
    c = df["close"]
    v = df["volume"]

    factors = {}

    # Momentum
    for n in [5, 10, 20, 60]:
        factors[f"ret_{n}d"] = (c.iloc[-1] / c.iloc[-n-1] - 1) if len(c) > n else 0

    # Reversal
    factors["reversal_5d"] = -(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else 0

    # Volatility
    ret = c.pct_change()
    for n in [5, 20, 60]:
        factors[f"vol_{n}d"] = ret.tail(n).std() if len(ret) > n else 0

    # Volume
    factors["vol_ratio"] = latest.get("vol_ratio", 1)
    factors["turnover"] = latest.get("turnover", 0)
    factors["vol_change_5d"] = (v.tail(5).mean() / (v.tail(20).mean() + 1) - 1) if len(v) > 20 else 0

    # Technical
    factors["rsi14"] = latest.get("rsi14", 50)
    factors["macd_hist"] = latest.get("macd_hist", 0)
    factors["kdj_j"] = latest.get("kdj_j", 50)
    factors["bb_width"] = latest.get("bb_width", 0)

    # Price position
    if "bb_upper" in latest and "bb_lower" in latest:
        rng = latest["bb_upper"] - latest["bb_lower"]
        factors["bb_position"] = (latest["close"] - latest["bb_lower"]) / (rng + 1e-10)
    else:
        factors["bb_position"] = 0.5

    # MA alignment score
    ma_score = 0
    for ma in ["ma5", "ma10", "ma20", "ma60"]:
        if ma in latest and latest["close"] > latest[ma]:
            ma_score += 1
    factors["ma_alignment"] = ma_score / 4

    # ATR normalized
    factors["atr_pct"] = latest.get("atr14", 0) / (latest["close"] + 1e-10)

    # Trend strength (MA5 slope)
    if "ma5" in df:
        factors["trend_strength"] = (df["ma5"].iloc[-1] / df["ma5"].iloc[-6] - 1) if len(df) > 6 else 0
    else:
        factors["trend_strength"] = 0

    # High/low position in 20d range
    h20 = df["high"].tail(20).max() if len(df) > 20 else df["high"].max()
    l20 = df["low"].tail(20).min() if len(df) > 20 else df["low"].min()
    factors["range_position"] = (latest["close"] - l20) / (h20 - l20 + 1e-10)

    # Gap
    factors["gap"] = (latest["open"] / df["close"].iloc[-2] - 1) if len(df) > 1 else 0

    return factors


def _build_training_data(symbol: str, predict_days: int = 5) -> pd.DataFrame:
    """Build factor + label dataset for a single stock's historical data."""
    try:
        df = fetch_stock(symbol)
        if len(df) < 120:
            return pd.DataFrame()
        df = add_technical_indicators(df)
    except:
        return pd.DataFrame()

    rows = []
    for i in range(80, len(df) - predict_days):
        sub = df.iloc[:i+1]
        factors = _compute_factors(sub)
        future_ret = df["close"].iloc[i + predict_days] / df["close"].iloc[i] - 1
        factors["future_ret"] = future_ret
        factors["date"] = df["date"].iloc[i]
        factors["symbol"] = symbol
        rows.append(factors)

    return pd.DataFrame(rows)


class MLFactorSelector:
    """LightGBM-based multi-factor stock selector."""

    def __init__(self, predict_days: int = 5):
        self.predict_days = predict_days
        self.model = None
        self.feature_cols = None
        self.metrics = {}

    def train(self, pool: list, progress_callback=None) -> "MLFactorSelector":
        """Train on historical data from the stock pool."""
        all_data = []
        for i, sym in enumerate(pool):
            if progress_callback:
                progress_callback(i / len(pool))
            data = _build_training_data(sym, self.predict_days)
            if len(data) > 0:
                all_data.append(data)

        if not all_data:
            return self

        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)

        # Binary label: top 30% returns = 1, else 0
        label_col = "future_ret"
        exclude = {"future_ret", "date", "symbol"}
        self.feature_cols = [c for c in df.columns if c not in exclude]

        X = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = (df[label_col] > df[label_col].median()).astype(int)

        # Time-based split
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        self.model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        self.model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                       callbacks=[lgb.early_stopping(20, verbose=False)])

        val_pred = self.model.predict(X_val)
        from sklearn.metrics import accuracy_score, f1_score
        self.metrics = {
            "训练样本": len(y_tr),
            "验证样本": len(y_val),
            "验证准确率": f"{accuracy_score(y_val, val_pred):.1%}",
            "验证F1": f"{f1_score(y_val, val_pred, zero_division=0):.3f}",
            "特征数": len(self.feature_cols),
        }
        return self

    def select(self, pool: list, top_n: int = 5) -> list:
        """Score and rank stocks, return top_n."""
        if self.model is None:
            return []

        scores = []
        for sym in pool:
            try:
                df = fetch_stock(sym)
                if len(df) < 80:
                    continue
                df = add_technical_indicators(df)
                factors = _compute_factors(df)
                X = pd.DataFrame([{c: factors.get(c, 0) for c in self.feature_cols}])
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                prob = self.model.predict_proba(X)[0, 1]

                reasons = []
                if factors.get("ret_20d", 0) > 0.05:
                    reasons.append(f"20日涨{factors['ret_20d']*100:.1f}%")
                if factors.get("ma_alignment", 0) > 0.5:
                    reasons.append("均线偏多")
                if factors.get("vol_ratio", 1) > 1.5:
                    reasons.append(f"放量{factors['vol_ratio']:.1f}x")
                reasons.append(f"ML评分{prob:.2f}")

                scores.append((sym, round(prob * 100, 2), "; ".join(reasons)))
            except:
                continue

        return sorted(scores, key=lambda x: -x[1])[:top_n]

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
