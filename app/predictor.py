"""
Prediction engine — train multiple models, generate forecasts and trading signals.

All models predict Δy(t) = yield(t) - yield(t-1), then reconstruct levels.
Returns predictions, direction probabilities, and confidence bands.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
import pmdarima as pm

warnings.filterwarnings("ignore")

LAGS = [1, 2, 3, 5, 10, 20, 60]


def _make_features(delta_series: pd.Series, level_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": delta_series.values}, index=delta_series.index)
    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_5"] = df["y"].shift(1).rolling(5).mean()
    df["roll_20"] = df["y"].shift(1).rolling(20).mean()
    df["roll_5_std"] = df["y"].shift(1).rolling(5).std()
    df["roll_20_std"] = df["y"].shift(1).rolling(20).std()
    df["diff2"] = df["y"].diff().shift(1)
    df["abs_lag1"] = df["lag_1"].abs()
    lvl = level_series.reindex(df.index)
    df["level_lag1"] = lvl.shift(1)
    df["level_ma20"] = lvl.shift(1).rolling(20).mean()
    return df.dropna()


def _build_models(X_train, y_train):
    models = {}

    # Ridge
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=alpha).fit(X_train, y_train)
        pred = m.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        if "Ridge" not in models or rmse < models["Ridge"]["train_rmse"]:
            models["Ridge"] = {"model": m, "train_rmse": rmse, "params": f"α={alpha}"}

    # XGBoost
    for md, lr, ne in [(3, 0.01, 500), (5, 0.05, 300), (3, 0.05, 500)]:
        m = xgb.XGBRegressor(max_depth=md, learning_rate=lr, n_estimators=ne,
                              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              random_state=42, verbosity=0).fit(X_train, y_train)
        pred = m.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        if "XGBoost" not in models or rmse < models["XGBoost"]["train_rmse"]:
            models["XGBoost"] = {"model": m, "train_rmse": rmse,
                                  "params": f"d={md},lr={lr},n={ne}"}

    # LightGBM
    for md, lr, nl in [(5, 0.01, 31), (3, 0.05, 31), (7, 0.01, 63)]:
        m = lgb.LGBMRegressor(max_depth=md, learning_rate=lr, n_estimators=500,
                               num_leaves=nl, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1).fit(X_train, y_train)
        pred = m.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, pred))
        if "LightGBM" not in models or rmse < models["LightGBM"]["train_rmse"]:
            models["LightGBM"] = {"model": m, "train_rmse": rmse,
                                   "params": f"d={md},lr={lr},nl={nl}"}

    # Random Forest
    m = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42,
                               n_jobs=-1).fit(X_train, y_train)
    models["RandomForest"] = {"model": m, "train_rmse": 0, "params": "n=300,d=10"}

    return models


class Predictor:
    def __init__(self, series: pd.Series, train_ratio: float = 0.85):
        self.raw = series.copy()
        self.raw.name = series.name or "asset"

        self.delta = series.diff().dropna()
        split = int(len(self.delta) * train_ratio)
        self.train_delta = self.delta.iloc[:split]
        self.test_delta = self.delta.iloc[split:]
        self.train_level = series.iloc[1:split+1]
        self.test_level = series.iloc[split+1:]
        self.prev_levels = series.iloc[split:-1].values if len(self.test_delta) > 0 else None

        feat = _make_features(self.delta, series.iloc[1:])
        self.feature_cols = [c for c in feat.columns if c != "y"]
        self.feat_train = feat.iloc[:split - max(LAGS) - 20]
        self.feat_test = feat.iloc[-(len(self.test_delta)):]

        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.feat_train[self.feature_cols]),
            index=self.feat_train.index, columns=self.feature_cols)
        self.y_train = self.feat_train["y"]

        if len(self.feat_test) > 0:
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.feat_test[self.feature_cols]),
                index=self.feat_test.index, columns=self.feature_cols)
            self.y_test = self.feat_test["y"]
        else:
            self.X_test = pd.DataFrame(columns=self.feature_cols)
            self.y_test = pd.Series(dtype=float)

        self.models = {}
        self.results = {}

    def train(self):
        self.models = _build_models(self.X_train, self.y_train)

        # ARIMA on delta
        try:
            arima = pm.auto_arima(self.train_delta, seasonal=False, stepwise=True,
                                   suppress_warnings=True, error_action="ignore",
                                   max_p=5, max_q=5, max_d=1)
            self.models["ARIMA"] = {"model": arima, "train_rmse": 0,
                                     "params": f"order={arima.order}"}
        except:
            pass

        return self

    def evaluate(self) -> pd.DataFrame:
        if len(self.X_test) == 0:
            return pd.DataFrame()

        rows = []
        test_len = len(self.y_test)
        prev = self.raw.iloc[-(test_len+1):-1].values

        for name, entry in self.models.items():
            mdl = entry["model"]
            if name == "ARIMA":
                delta_pred = mdl.predict(n_periods=test_len)
            else:
                delta_pred = mdl.predict(self.X_test[:test_len])

            lvl_pred = prev + delta_pred
            lvl_true = self.test_level.values[:test_len]
            delta_true = self.y_test.values[:test_len]

            d_rmse = np.sqrt(mean_squared_error(delta_true, delta_pred))
            l_rmse = np.sqrt(mean_squared_error(lvl_true, lvl_pred))
            dir_acc = ((delta_true > 0) == (delta_pred > 0)).mean() * 100

            rows.append({
                "模型": name, "Δ-RMSE": round(d_rmse, 6),
                "Level-RMSE": round(l_rmse, 6),
                "方向准确率(%)": round(dir_acc, 1),
                "参数": entry["params"],
            })
            self.results[name] = {
                "delta_pred": delta_pred, "lvl_pred": lvl_pred,
                "delta_true": delta_true, "lvl_true": lvl_true,
                "dir_acc": dir_acc, "d_rmse": d_rmse,
            }

        return pd.DataFrame(rows).sort_values("Δ-RMSE").reset_index(drop=True)

    def predict_next(self, n_days: int = 5) -> pd.DataFrame:
        """Predict next n_days from the latest data."""
        predictions = {}
        last_level = self.raw.iloc[-1]

        for name, entry in self.models.items():
            mdl = entry["model"]
            if name == "ARIMA":
                delta_preds = mdl.predict(n_periods=n_days)
            else:
                preds = []
                current_delta = self.delta.copy()
                current_level = self.raw.copy()
                for step in range(n_days):
                    feat = _make_features(current_delta, current_level.iloc[1:])
                    last_row = feat.iloc[[-1]][self.feature_cols]
                    last_scaled = self.scaler.transform(last_row)
                    d_pred = mdl.predict(last_scaled)[0]
                    preds.append(d_pred)
                    new_level = current_level.iloc[-1] + d_pred
                    new_date = current_level.index[-1] + pd.tseries.offsets.BDay(1)
                    current_level = pd.concat([current_level, pd.Series([new_level], index=[new_date])])
                    current_delta = pd.concat([current_delta, pd.Series([d_pred], index=[new_date])])
                delta_preds = np.array(preds)

            levels = np.cumsum(delta_preds) + last_level
            predictions[name] = {"delta": delta_preds, "level": levels}

        # Ensemble (average of all models)
        all_deltas = np.array([v["delta"] for v in predictions.values()])
        ens_delta = all_deltas.mean(axis=0)
        ens_level = np.cumsum(ens_delta) + last_level
        predictions["集成(均值)"] = {"delta": ens_delta, "level": ens_level}

        # Build output table
        future_dates = pd.bdate_range(self.raw.index[-1] + pd.tseries.offsets.BDay(1), periods=n_days)
        rows = []
        for i, d in enumerate(future_dates):
            row = {"日期": d.strftime("%Y-%m-%d")}
            for name, v in predictions.items():
                row[f"{name}_预测"] = round(v["level"][i], 4)
                row[f"{name}_Δ"] = round(v["delta"][i], 4)
            rows.append(row)

        return pd.DataFrame(rows), predictions

    def generate_signals(self, predictions: dict, threshold: float = 0.002) -> dict:
        """Generate trading signals based on ensemble prediction."""
        ens = predictions.get("集成(均值)", {})
        if not ens:
            return {}

        delta_sum = ens["delta"].sum()
        direction = "看多 📈" if delta_sum > threshold else ("看空 📉" if delta_sum < -threshold else "观望 ➡️")

        # Confidence: agreement ratio among models
        model_dirs = []
        for name, v in predictions.items():
            if name == "集成(均值)":
                continue
            model_dirs.append(1 if v["delta"].sum() > 0 else -1)
        agreement = abs(sum(model_dirs)) / len(model_dirs) if model_dirs else 0

        if agreement > 0.8:
            confidence = "高"
        elif agreement > 0.5:
            confidence = "中"
        else:
            confidence = "低"

        return {
            "方向": direction,
            "预期变化": round(delta_sum, 4),
            "模型一致性": f"{agreement:.0%}",
            "信心水平": confidence,
            "看多模型": sum(1 for d in model_dirs if d > 0),
            "看空模型": sum(1 for d in model_dirs if d < 0),
        }
