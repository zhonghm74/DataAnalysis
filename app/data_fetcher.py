"""
Data fetcher module — download and cache financial time series from akshare.

Supports:
  - Chinese government bonds (2Y, 5Y, 10Y, 30Y)
  - US government bonds (2Y, 5Y, 10Y, 30Y)
"""

import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ASSET_MAP = {
    "中国国债2年": "中国国债收益率2年",
    "中国国债5年": "中国国债收益率5年",
    "中国国债10年": "中国国债收益率10年",
    "中国国债30年": "中国国债收益率30年",
    "美国国债2年": "美国国债收益率2年",
    "美国国债5年": "美国国债收益率5年",
    "美国国债10年": "美国国债收益率10年",
    "美国国债30年": "美国国债收益率30年",
}


def fetch_bond_data(start_date: str = "20150101") -> pd.DataFrame:
    cache_path = os.path.join(CACHE_DIR, "bond_rates.csv")
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=12):
            return pd.read_csv(cache_path, parse_dates=["日期"])

    df = ak.bond_zh_us_rate(start_date=start_date)
    df["日期"] = pd.to_datetime(df["日期"])
    df.to_csv(cache_path, index=False)
    return df


def get_series(asset_name: str, start_date: str = "20150101") -> pd.Series:
    col = ASSET_MAP.get(asset_name)
    if col is None:
        raise ValueError(f"Unknown asset: {asset_name}. Choose from {list(ASSET_MAP.keys())}")
    df = fetch_bond_data(start_date)
    s = df.set_index("日期")[col].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("B").ffill().dropna()
    s.name = asset_name
    return s


def list_assets():
    return list(ASSET_MAP.keys())
