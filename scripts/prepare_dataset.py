"""
Feature engineering, variable selection, transformations, sample balancing,
and train/test split for the credit‑card fraud detection dataset.

Outputs (→ data/processed/):
  - X_train.csv / y_train.csv          原始训练集
  - X_test.csv  / y_test.csv           测试集
  - X_train_smote.csv / y_train_smote.csv  SMOTE 均衡后的训练集
  - feature_metadata.csv               特征元信息
  - preparation_report.md              数据准备报告（→ reports/）
"""

import os, sys, warnings, json
from datetime import datetime
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "fraudTrain.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
REPORT_PATH = os.path.join(ROOT, "reports", "data_preparation_report.md")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"

# ===================================================================
# 1. LOAD RAW DATA
# ===================================================================
print("=" * 60)
print("Step 1: Loading raw data …")
df = pd.read_csv(DATA_PATH)
if df.columns[0] in ("Unnamed: 0", ""):
    df.rename(columns={df.columns[0]: "row_index"}, inplace=True)
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
print(f"  Raw shape: {df.shape}")

TARGET = "is_fraud"
R = []  # report lines

# ===================================================================
# 2. FEATURE ENGINEERING — DERIVED VARIABLES
# ===================================================================
print("\nStep 2: Feature engineering — derived variables …")

# 2.1 Time features
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
df["trans_month"] = df["trans_date_trans_time"].dt.month
df["trans_day"] = df["trans_date_trans_time"].dt.day
df["is_weekend"] = (df["trans_dayofweek"] >= 5).astype(int)
df["is_night"] = ((df["trans_hour"] >= 22) | (df["trans_hour"] <= 5)).astype(int)

# 2.2 Age
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25

# 2.3 Geographic distance (Haversine approximation in km)
def haversine_km(lat1, lon1, lat2, lon2):
    R_EARTH = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine_km(
    df["lat"].values, df["long"].values,
    df["merch_lat"].values, df["merch_long"].values
)

# 2.4 Amount transformations
df["amt_log"] = np.log1p(df["amt"])

# 2.5 City population log
df["city_pop_log"] = np.log1p(df["city_pop"])

# 2.6 Hour cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["trans_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["trans_hour"] / 24)

# 2.7 Day-of-week cyclical encoding
df["dow_sin"] = np.sin(2 * np.pi * df["trans_dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["trans_dayofweek"] / 7)

# 2.8 Category frequency encoding
cat_freq = df["category"].value_counts(normalize=True)
df["category_freq"] = df["category"].map(cat_freq)

# 2.9 Merchant frequency encoding
merch_freq = df["merchant"].value_counts(normalize=True)
df["merchant_freq"] = df["merchant"].map(merch_freq)

# 2.10 State frequency encoding
state_freq = df["state"].value_counts(normalize=True)
df["state_freq"] = df["state"].map(state_freq)

# 2.11 Job frequency encoding
job_freq = df["job"].value_counts(normalize=True)
df["job_freq"] = df["job"].map(job_freq)

# 2.12 Category label encoding
le_category = LabelEncoder()
df["category_enc"] = le_category.fit_transform(df["category"])

# 2.13 Gender encoding
df["gender_enc"] = (df["gender"] == "M").astype(int)

derived_features = [
    ("trans_hour",     "时间特征", "交易发生的小时 (0-23)"),
    ("trans_dayofweek","时间特征", "交易发生的星期几 (0=Mon, 6=Sun)"),
    ("trans_month",    "时间特征", "交易发生的月份 (1-12)"),
    ("trans_day",      "时间特征", "交易发生的日期 (1-31)"),
    ("is_weekend",     "时间特征", "是否周末 (0/1)"),
    ("is_night",       "时间特征", "是否夜间 22:00-05:00 (0/1)"),
    ("hour_sin",       "时间特征（周期编码）", "小时正弦编码"),
    ("hour_cos",       "时间特征（周期编码）", "小时余弦编码"),
    ("dow_sin",        "时间特征（周期编码）", "星期正弦编码"),
    ("dow_cos",        "时间特征（周期编码）", "星期余弦编码"),
    ("age",            "人口统计", "持卡人年龄（由 dob 计算）"),
    ("distance_km",    "地理特征", "持卡人与商户之间的 Haversine 距离 (km)"),
    ("amt_log",        "金额变换", "log(1 + amt)，缓解右偏"),
    ("city_pop_log",   "人口变换", "log(1 + city_pop)，缓解右偏"),
    ("category_freq",  "频率编码", "消费类别出现频率"),
    ("merchant_freq",  "频率编码", "商户出现频率"),
    ("state_freq",     "频率编码", "州出现频率"),
    ("job_freq",       "频率编码", "职业出现频率"),
    ("category_enc",   "标签编码", "消费类别的 Label Encoding"),
    ("gender_enc",     "二值编码", "性别 (M=1, F=0)"),
]

print(f"  Created {len(derived_features)} derived features")

# ===================================================================
# 3. VARIABLE SELECTION
# ===================================================================
print("\nStep 3: Variable selection …")

DROP_COLS = [
    "row_index",          # ID
    "trans_num",          # ID
    "cc_num",             # ID
    "unix_time",          # ID (信息已被 time features 覆盖)
    "trans_date_trans_time",  # 已拆解为时间特征
    "dob",                # 已转化为 age
    "first",              # 个人信息，无预测价值
    "last",               # 个人信息
    "street",             # 地址文本，高基数低信息量
    "merchant",           # 已通过 merchant_freq 编码
    "category",           # 已通过 category_enc / category_freq 编码
    "gender",             # 已通过 gender_enc 编码
    "city",               # 高基数，信息量低
    "state",              # 已通过 state_freq 编码
    "job",                # 已通过 job_freq 编码
    "merch_lat",          # 与 lat 高度相关 (r=0.99)，用 distance_km 替代
    "merch_long",         # 与 long 高度相关 (r=0.99)，用 distance_km 替代
    "trans_hour",         # 已通过 hour_sin/cos 周期编码
    "trans_dayofweek",    # 已通过 dow_sin/cos 周期编码
]

drop_reasons = {
    "row_index": "ID 列，无预测价值",
    "trans_num": "ID 列，无预测价值",
    "cc_num": "ID 列，无预测价值",
    "unix_time": "ID 列，时间信息已由派生特征覆盖",
    "trans_date_trans_time": "时间戳已拆解为多个时间特征",
    "dob": "已转化为 age 特征",
    "first": "个人信息文本，高基数无预测价值",
    "last": "个人信息文本，高基数无预测价值",
    "street": "地址文本，高基数低信息量",
    "merchant": "已通过 merchant_freq 频率编码",
    "category": "已通过 category_enc + category_freq 编码",
    "gender": "已通过 gender_enc 二值编码",
    "city": "高基数 (894 值)，信息量低",
    "state": "已通过 state_freq 频率编码",
    "job": "已通过 job_freq 频率编码",
    "merch_lat": "与 lat 高度相关 (r=0.99)，用 distance_km 替代",
    "merch_long": "与 long 高度相关 (r=0.99)，用 distance_km 替代",
    "trans_hour": "已通过 hour_sin/cos 周期编码替代",
    "trans_dayofweek": "已通过 dow_sin/cos 周期编码替代",
}

existing_drops = [c for c in DROP_COLS if c in df.columns]
df_model = df.drop(columns=existing_drops + [TARGET])
feature_names = df_model.columns.tolist()
y_full = df[TARGET].values

print(f"  Dropped {len(existing_drops)} columns")
print(f"  Remaining features: {len(feature_names)}")
print(f"  Features: {feature_names}")

# ===================================================================
# 4. TRAIN / TEST SPLIT (time‑based)
# ===================================================================
print("\nStep 4: Train / test split (time‑based) …")

df_sorted = df.sort_values("trans_date_trans_time").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)
split_date = df_sorted.iloc[split_idx]["trans_date_trans_time"]

train_mask = df["trans_date_trans_time"] < split_date
test_mask = ~train_mask

X_train_raw = df_model.loc[train_mask].copy()
X_test_raw = df_model.loc[test_mask].copy()
y_train = df.loc[train_mask, TARGET].values
y_test = df.loc[test_mask, TARGET].values

print(f"  Split date: {split_date}")
print(f"  Train: {X_train_raw.shape[0]:,} rows  |  Test: {X_test_raw.shape[0]:,} rows")
print(f"  Train fraud rate: {y_train.mean():.4f}  |  Test fraud rate: {y_test.mean():.4f}")

# ===================================================================
# 5. NUMERICAL TRANSFORMATIONS (RobustScaler)
# ===================================================================
print("\nStep 5: Numerical scaling (RobustScaler) …")

numeric_features = X_train_raw.select_dtypes(include="number").columns.tolist()
scaler = RobustScaler()
X_train_scaled = X_train_raw.copy()
X_test_scaled = X_test_raw.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train_raw[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_raw[numeric_features])

print(f"  Scaled {len(numeric_features)} numerical features with RobustScaler")

# ===================================================================
# 6. FEATURE IMPORTANCE — Mutual Information
# ===================================================================
print("\nStep 6: Feature importance (Mutual Information) …")

mi_scores = mutual_info_classif(
    X_train_scaled.fillna(0), y_train,
    discrete_features="auto", random_state=42, n_neighbors=5
)
mi_df = pd.DataFrame({
    "feature": feature_names,
    "MI_score": mi_scores
}).sort_values("MI_score", ascending=False).reset_index(drop=True)
mi_df["rank"] = range(1, len(mi_df) + 1)

print("  Top 10 features by MI:")
for _, row in mi_df.head(10).iterrows():
    print(f"    {row['rank']:2d}. {row['feature']:20s}  MI={row['MI_score']:.4f}")

LOW_MI_THRESHOLD = 0.001
low_mi = mi_df[mi_df["MI_score"] < LOW_MI_THRESHOLD]["feature"].tolist()
if low_mi:
    print(f"\n  Dropping {len(low_mi)} features with MI < {LOW_MI_THRESHOLD}: {low_mi}")
    X_train_scaled.drop(columns=low_mi, inplace=True)
    X_test_scaled.drop(columns=low_mi, inplace=True)
    X_train_raw.drop(columns=low_mi, inplace=True)
    X_test_raw.drop(columns=low_mi, inplace=True)
    feature_names = [f for f in feature_names if f not in low_mi]

print(f"  Final feature count: {len(feature_names)}")

# ===================================================================
# 7. SMOTE OVERSAMPLING (train only)
# ===================================================================
print("\nStep 7: SMOTE oversampling on training set …")

smote = SMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"  Before SMOTE: {len(y_train):,}  (fraud={y_train.sum():,}, normal={(y_train==0).sum():,})")
print(f"  After  SMOTE: {len(y_train_smote):,}  (fraud={(y_train_smote==1).sum():,}, normal={(y_train_smote==0).sum():,})")

# ===================================================================
# 8. SAVE DATASETS
# ===================================================================
print("\nStep 8: Saving processed datasets …")

X_train_scaled.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
pd.Series(y_train, name=TARGET).to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
pd.Series(y_test, name=TARGET).to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)
X_train_smote.to_csv(os.path.join(OUT_DIR, "X_train_smote.csv"), index=False)
pd.Series(y_train_smote, name=TARGET).to_csv(os.path.join(OUT_DIR, "y_train_smote.csv"), index=False)
mi_df.to_csv(os.path.join(OUT_DIR, "feature_metadata.csv"), index=False)

# Save scaler params
import pickle
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
# Save label encoder
with open(os.path.join(OUT_DIR, "le_category.pkl"), "wb") as f:
    pickle.dump(le_category, f)

for fname in os.listdir(OUT_DIR):
    size = os.path.getsize(os.path.join(OUT_DIR, fname))
    print(f"  {fname:30s}  {size/1024/1024:.1f} MB")

# ===================================================================
# 9. GENERATE REPORT & FIGURES
# ===================================================================
print("\nStep 9: Generating data preparation report …")

# --- Fig: MI bar chart ---
fig, ax = plt.subplots(figsize=(10, 8))
mi_plot = mi_df.sort_values("MI_score", ascending=True)
colors = ["crimson" if s > 0.01 else "steelblue" for s in mi_plot["MI_score"]]
ax.barh(mi_plot["feature"], mi_plot["MI_score"], color=colors)
ax.set_xlabel("Mutual Information Score")
ax.set_title("特征重要性 — Mutual Information with Target", fontsize=13)
ax.axvline(LOW_MI_THRESHOLD, color="gray", linestyle="--", alpha=0.7, label=f"阈值={LOW_MI_THRESHOLD}")
ax.legend()
fig.tight_layout()
p_mi = savefig(fig, "feature_importance_mi.png")

# --- Fig: Class balance before/after SMOTE ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ["正常 (0)", "欺诈 (1)"]

# Before
counts_before = [int((y_train == 0).sum()), int((y_train == 1).sum())]
axes[0].bar(labels, counts_before, color=["steelblue", "crimson"], edgecolor="white")
axes[0].set_title("SMOTE 前（训练集）", fontsize=12)
axes[0].set_ylabel("样本数")
for i, v in enumerate(counts_before):
    axes[0].text(i, v + max(counts_before) * 0.02, f"{v:,}", ha="center")

# After
counts_after = [int((y_train_smote == 0).sum()), int((y_train_smote == 1).sum())]
axes[1].bar(labels, counts_after, color=["steelblue", "crimson"], edgecolor="white")
axes[1].set_title("SMOTE 后（训练集）", fontsize=12)
axes[1].set_ylabel("样本数")
for i, v in enumerate(counts_after):
    axes[1].text(i, v + max(counts_after) * 0.02, f"{v:,}", ha="center")

fig.suptitle("样本均衡：SMOTE 前后对比", fontsize=14, y=1.02)
fig.tight_layout()
p_balance = savefig(fig, "smote_balance.png")

# --- Fig: amt_log & distance_km distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df["amt"], bins=80, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_title("amt 原始分布", fontsize=12)
axes[0].set_xlabel("amt")
axes[0].set_xlim(0, df["amt"].quantile(0.99))

axes[1].hist(df["amt_log"], bins=80, color="teal", edgecolor="white", alpha=0.8)
axes[1].set_title("amt_log = log(1+amt) 变换后", fontsize=12)
axes[1].set_xlabel("amt_log")

axes[2].hist(df["distance_km"].clip(upper=df["distance_km"].quantile(0.99)),
             bins=80, color="darkorange", edgecolor="white", alpha=0.8)
axes[2].set_title("distance_km 分布（持卡人-商户距离）", fontsize=12)
axes[2].set_xlabel("km")

fig.suptitle("关键衍生变量分布", fontsize=14, y=1.02)
fig.tight_layout()
p_derived = savefig(fig, "derived_features_dist.png")

# --- Fig: Correlation heatmap (final features) ---
fig, ax = plt.subplots(figsize=(14, 12))
corr_final = X_train_scaled.corr()
mask = np.triu(np.ones_like(corr_final, dtype=bool), k=1)
sns.heatmap(corr_final, mask=mask, cmap="RdBu_r", center=0, square=True,
            linewidths=0.3, ax=ax, vmin=-1, vmax=1,
            annot=True, fmt=".1f", annot_kws={"size": 7})
ax.set_title("最终特征相关性矩阵", fontsize=14)
fig.tight_layout()
p_corr = savefig(fig, "final_feature_correlation.png")

# --- Fig: distance_km by fraud ---
fig, ax = plt.subplots(figsize=(10, 5))
df_vis = df.copy()
df_vis["is_fraud_label"] = df_vis[TARGET].map({0: "正常", 1: "欺诈"})
for label, color in [("正常", "steelblue"), ("欺诈", "crimson")]:
    subset = df_vis[df_vis["is_fraud_label"] == label]["distance_km"]
    ax.hist(subset.clip(upper=subset.quantile(0.99)), bins=60, alpha=0.6,
            color=color, edgecolor="white", label=label, density=True)
ax.set_title("distance_km 分布：正常 vs 欺诈", fontsize=13)
ax.set_xlabel("距离 (km)")
ax.set_ylabel("密度")
ax.legend()
fig.tight_layout()
p_dist_fraud = savefig(fig, "distance_by_fraud.png")

# ===================================================================
# BUILD MARKDOWN REPORT
# ===================================================================
R.append("# 数据准备与特征工程报告\n")
R.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
R.append("---\n")

# --- Section 1: Overview ---
R.append("## 1. 概述\n")
R.append(dedent(f"""\
本报告描述了 `fraudTrain.csv` 从原始数据到建模可用数据集的完整处理流程，包括：
- 衍生变量构造
- 变量筛选与剔除
- 数值变换与缩放
- 特征重要性评估
- 训练集/测试集划分
- SMOTE 样本均衡
"""))

# --- Section 2: Derived features ---
R.append("## 2. 衍生变量\n")
R.append(f"共构造 **{len(derived_features)}** 个衍生特征：\n")
R.append("| 特征 | 类别 | 说明 |")
R.append("|---|---|---|")
for fname, fcat, fdesc in derived_features:
    R.append(f"| `{fname}` | {fcat} | {fdesc} |")
R.append("")
R.append(f"![衍生变量分布]({p_derived})\n")
R.append(f"![距离与欺诈关系]({p_dist_fraud})\n")

# --- Section 3: Dropped columns ---
R.append("## 3. 剔除变量\n")
R.append(f"共剔除 **{len(existing_drops)}** 个原始列：\n")
R.append("| 剔除列 | 原因 |")
R.append("|---|---|")
for c in existing_drops:
    R.append(f"| `{c}` | {drop_reasons.get(c, '')} |")
R.append("")

# --- Section 4: Transformations ---
R.append("## 4. 数值变换\n")
R.append(dedent("""\
| 变换 | 目标列 | 方法 | 说明 |
|---|---|---|---|
| 对数变换 | `amt` → `amt_log` | `log(1+x)` | 缓解交易金额严重右偏 (skew=42.3) |
| 对数变换 | `city_pop` → `city_pop_log` | `log(1+x)` | 缓解城市人口严重右偏 (skew=5.6) |
| 周期编码 | `trans_hour` → `hour_sin/cos` | 正弦/余弦 | 保留小时的周期性（23→0 连续） |
| 周期编码 | `trans_dayofweek` → `dow_sin/cos` | 正弦/余弦 | 保留星期的周期性 |
| 频率编码 | `category/merchant/state/job` | 频率占比 | 高基数分类变量 → 连续值 |
| 全局缩放 | 所有数值特征 | RobustScaler | 基于中位数和 IQR，对异常值鲁棒 |
"""))

# --- Section 5: Feature importance ---
R.append("## 5. 特征重要性（Mutual Information）\n")
R.append(f"![特征重要性]({p_mi})\n")
R.append("| 排名 | 特征 | MI Score |")
R.append("|---|---|---|")
for _, row in mi_df.iterrows():
    marker = " ⚠️" if row["MI_score"] < LOW_MI_THRESHOLD else ""
    R.append(f"| {int(row['rank'])} | `{row['feature']}` | {row['MI_score']:.4f}{marker} |")
R.append("")
if low_mi:
    R.append(f"> ⚠️ 已剔除 MI < {LOW_MI_THRESHOLD} 的低信息特征: {', '.join([f'`{c}`' for c in low_mi])}\n")

# --- Section 6: Train / Test Split ---
R.append("## 6. 训练集 / 测试集划分\n")
R.append(dedent(f"""\
采用 **基于时间的划分** (time‑based split)，避免数据泄漏：
- **划分日期**: `{split_date}`
- **划分比例**: 前 80% 为训练集，后 20% 为测试集

| 数据集 | 样本数 | 欺诈数 | 欺诈率 |
|---|---|---|---|
| 训练集 | {len(y_train):,} | {int(y_train.sum()):,} | {y_train.mean()*100:.2f}% |
| 测试集 | {len(y_test):,} | {int(y_test.sum()):,} | {y_test.mean()*100:.2f}% |
"""))

# --- Section 7: SMOTE ---
R.append("## 7. 样本均衡 — SMOTE\n")
R.append(dedent(f"""\
对训练集使用 **SMOTE (Synthetic Minority Over-sampling Technique)** 进行过采样：
- 算法: SMOTE (k_neighbors=5)
- 策略: sampling_strategy=1.0（使正负样本数量相等）

| 阶段 | 正常样本 | 欺诈样本 | 总计 | 欺诈比例 |
|---|---|---|---|---|
| SMOTE 前 | {counts_before[0]:,} | {counts_before[1]:,} | {sum(counts_before):,} | {counts_before[1]/sum(counts_before)*100:.2f}% |
| SMOTE 后 | {counts_after[0]:,} | {counts_after[1]:,} | {sum(counts_after):,} | {counts_after[1]/sum(counts_after)*100:.2f}% |

> **注意**: SMOTE 仅应用于训练集，测试集保持原始分布以反映真实场景。
"""))
R.append(f"![SMOTE 前后对比]({p_balance})\n")

# --- Section 8: Final feature set ---
R.append("## 8. 最终特征集\n")
R.append(f"最终进入模型的特征共 **{len(feature_names)}** 个：\n")
R.append(f"![最终特征相关性]({p_corr})\n")
R.append("```")
R.append(", ".join(feature_names))
R.append("```\n")

# --- Section 9: Output files ---
R.append("## 9. 输出文件\n")
R.append("| 文件 | 说明 |")
R.append("|---|---|")
R.append("| `data/processed/X_train.csv` | 训练集特征（缩放后，原始比例） |")
R.append("| `data/processed/y_train.csv` | 训练集标签 |")
R.append("| `data/processed/X_test.csv` | 测试集特征（缩放后） |")
R.append("| `data/processed/y_test.csv` | 测试集标签 |")
R.append("| `data/processed/X_train_smote.csv` | 训练集特征（SMOTE 均衡后） |")
R.append("| `data/processed/y_train_smote.csv` | 训练集标签（SMOTE 均衡后） |")
R.append("| `data/processed/feature_metadata.csv` | 特征 MI 重要性排名 |")
R.append("| `data/processed/scaler.pkl` | RobustScaler 拟合参数 |")
R.append("| `data/processed/le_category.pkl` | 类别 LabelEncoder |")
R.append("")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n{'='*60}")
print(f"✅ Data preparation complete!")
print(f"   Report:   {REPORT_PATH}")
print(f"   Datasets: {OUT_DIR}/")
print(f"   Features: {len(feature_names)}")
print(f"   Train:    {X_train_scaled.shape}  →  SMOTE: {X_train_smote.shape}")
print(f"   Test:     {X_test_scaled.shape}")
