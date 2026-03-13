"""
Comprehensive EDA for fraudTrain.csv — Credit Card Fraud Detection Dataset.

Generates a Markdown report with embedded figures covering:
  1. Dataset overview & dimension descriptions
  2. Missing values analysis
  3. Outlier detection (IQR method)
  4. Distribution analysis (numerical + categorical)
  5. Correlation analysis & hierarchical clustering
  6. Feature–target relationship analysis
  7. Target class balance & missing‑value summary
  8. Modeling recommendations
"""

import os
import warnings
from datetime import datetime
from textwrap import dedent

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import plot_config  # Chinese font support — must be before plt/sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fraudTrain.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
REPORT_PATH = os.path.join(REPORT_DIR, "fraudTrain_eda_report.md")

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def savefig(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"

def md_table(df_: pd.DataFrame) -> str:
    return df_.to_markdown(index=False)

def pct(x, total):
    return f"{x} ({x / total * 100:.2f}%)"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(DATA_PATH)
if df.columns[0] == "Unnamed: 0" or df.columns[0] == "":
    df = df.rename(columns={df.columns[0]: "row_index"})

n_rows, n_cols = df.shape
print(f"  Loaded {n_rows:,} rows × {n_cols} columns")

# Classify columns
TARGET = "is_fraud"
ID_COLS = ["row_index", "trans_num", "cc_num", "unix_time"]
TIME_COLS = ["trans_date_trans_time", "dob"]
NUM_COLS = [c for c in df.select_dtypes(include="number").columns if c not in ID_COLS + [TARGET]]
CAT_COLS = [c for c in df.select_dtypes(include="object").columns if c not in TIME_COLS + ID_COLS]

# Parse time
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
NUM_COLS.append("age")

# Derived features for analysis
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek

# ---------------------------------------------------------------------------
# Column descriptions
# ---------------------------------------------------------------------------
COL_DESC = {
    "row_index": ("ID", "原始行索引"),
    "trans_date_trans_time": ("时间", "交易发生的日期和时间"),
    "cc_num": ("ID", "信用卡号（脱敏）"),
    "merchant": ("分类", "商户名称"),
    "category": ("分类", "消费类别（如 grocery_pos, shopping_net 等）"),
    "amt": ("数值", "交易金额（美元）"),
    "first": ("分类", "持卡人名"),
    "last": ("分类", "持卡人姓"),
    "gender": ("分类", "持卡人性别（M/F）"),
    "street": ("分类", "持卡人街道地址"),
    "city": ("分类", "持卡人所在城市"),
    "state": ("分类", "持卡人所在州"),
    "zip": ("数值", "持卡人邮编"),
    "lat": ("数值", "交易纬度"),
    "long": ("数值", "交易经度"),
    "city_pop": ("数值", "城市人口"),
    "job": ("分类", "持卡人职业"),
    "dob": ("时间", "持卡人出生日期"),
    "trans_num": ("ID", "交易唯一编号"),
    "unix_time": ("ID", "交易的 Unix 时间戳"),
    "merch_lat": ("数值", "商户纬度"),
    "merch_long": ("数值", "商户经度"),
    "is_fraud": ("目标", "是否欺诈（0=正常，1=欺诈）"),
    "age": ("数值（派生）", "持卡人年龄（由 dob 计算）"),
}

# ===================================================================
# START BUILDING REPORT
# ===================================================================
R = []  # report lines

R.append("# fraudTrain.csv 探索性数据分析 (EDA) 报告\n")
R.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
R.append("---\n")

# ---------------------------------------------------------------------------
# Section 1 — Dataset Overview
# ---------------------------------------------------------------------------
R.append("## 1. 数据集概览\n")
R.append(f"- **来源**: Kaggle `kartik2112/fraud-detection`")
R.append(f"- **文件**: `data/fraudTrain.csv`")
R.append(f"- **总行数**: {n_rows:,}")
R.append(f"- **总列数**: {n_cols}")
R.append(f"- **目标变量**: `is_fraud`（二分类：0=正常, 1=欺诈）\n")

# ---------------------------------------------------------------------------
# Section 2 — Column Descriptions
# ---------------------------------------------------------------------------
R.append("## 2. 各维度说明\n")
desc_rows = []
for c in df.columns:
    if c in ["trans_hour", "trans_dayofweek"]:
        continue
    dtype = str(df[c].dtype)
    ctype, cdesc = COL_DESC.get(c, ("未知", ""))
    desc_rows.append({"列名": c, "数据类型": dtype, "维度类别": ctype, "说明": cdesc})
desc_df = pd.DataFrame(desc_rows)
R.append(md_table(desc_df))
R.append("")

# ---------------------------------------------------------------------------
# Section 3 — Missing Values
# ---------------------------------------------------------------------------
R.append("## 3. 缺失值分析\n")
missing = df.isnull().sum()
missing_pct = (missing / n_rows * 100).round(4)
miss_df = pd.DataFrame({"列名": missing.index, "缺失数": missing.values, "缺失率(%)": missing_pct.values})
miss_df = miss_df.sort_values("缺失数", ascending=False).reset_index(drop=True)
total_missing = missing.sum()

if total_missing == 0:
    R.append("**所有列均无缺失值。** 数据完整性良好。\n")
else:
    R.append(f"共有 **{total_missing:,}** 个缺失值，分布如下：\n")
R.append(md_table(miss_df[miss_df["缺失数"] > 0]) if total_missing > 0 else md_table(miss_df.head(10)))
R.append("")

# Missing value heatmap (sample for performance)
print("Generating missing value heatmap …")
fig, ax = plt.subplots(figsize=(14, 4))
sample_idx = np.random.choice(n_rows, size=min(5000, n_rows), replace=False)
sns.heatmap(df.iloc[sample_idx].isnull().T, cbar=False, yticklabels=True, cmap="YlOrRd", ax=ax)
ax.set_title("缺失值热力图（随机采样 5000 行）", fontsize=13)
ax.set_xlabel("样本")
p = savefig(fig, "missing_heatmap.png")
R.append(f"![缺失值热力图]({p})\n")

# ---------------------------------------------------------------------------
# Section 4 — Outlier Analysis
# ---------------------------------------------------------------------------
R.append("## 4. 异常值分析（IQR 方法）\n")
R.append("使用 1.5×IQR 规则检测数值型特征的异常值：\n")

outlier_rows = []
for c in NUM_COLS:
    s = df[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < lower) | (s > upper)).sum()
    outlier_rows.append({
        "特征": c, "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2),
        "下界": round(lower, 2), "上界": round(upper, 2),
        "异常值数": n_out, "异常值占比(%)": round(n_out / len(s) * 100, 2),
    })
out_df = pd.DataFrame(outlier_rows).sort_values("异常值占比(%)", ascending=False).reset_index(drop=True)
R.append(md_table(out_df))
R.append("")

# Box plots for key numerical features
print("Generating outlier box plots …")
key_num = ["amt", "city_pop", "age", "lat", "long", "merch_lat", "merch_long", "zip"]
key_num = [c for c in key_num if c in NUM_COLS]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for i, c in enumerate(key_num):
    sns.boxplot(y=df[c], ax=axes[i], color="skyblue", fliersize=1)
    axes[i].set_title(c, fontsize=11)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("数值特征箱线图（异常值检测）", fontsize=14, y=1.02)
fig.tight_layout()
p = savefig(fig, "outlier_boxplots.png")
R.append(f"![箱线图]({p})\n")

# ---------------------------------------------------------------------------
# Section 5 — Distribution Analysis
# ---------------------------------------------------------------------------
R.append("## 5. 数据分布分析\n")

# 5.1 Numerical distributions
R.append("### 5.1 数值特征分布\n")

desc_stat = df[NUM_COLS].describe().T
desc_stat["skew"] = df[NUM_COLS].skew()
desc_stat["kurtosis"] = df[NUM_COLS].kurtosis()
desc_stat = desc_stat.round(4)
desc_stat.index.name = "特征"
R.append(md_table(desc_stat.reset_index()))
R.append("")

print("Generating numerical distribution plots …")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()
plot_cols = ["amt", "city_pop", "age", "lat", "long", "merch_lat", "merch_long", "zip"]
plot_cols = [c for c in plot_cols if c in NUM_COLS]
for i, c in enumerate(plot_cols):
    if i >= len(axes):
        break
    ax = axes[i]
    data_c = df[c].dropna()
    if c == "amt":
        data_c = data_c[data_c < data_c.quantile(0.99)]
    elif c == "city_pop":
        data_c = data_c[data_c < data_c.quantile(0.99)]
    ax.hist(data_c, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(f"{c}  (skew={df[c].skew():.2f})", fontsize=11)
    ax.set_ylabel("频次")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("数值特征分布直方图", fontsize=14, y=1.01)
fig.tight_layout()
p = savefig(fig, "numerical_distributions.png")
R.append(f"![数值分布]({p})\n")

# AMT distribution by fraud
print("Generating amt distribution by fraud …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color, ax_i in [(0, "steelblue", 0), (1, "crimson", 1)]:
    subset = df[df[TARGET] == label]["amt"]
    title = "正常交易 (is_fraud=0)" if label == 0 else "欺诈交易 (is_fraud=1)"
    axes[ax_i].hist(subset.clip(upper=subset.quantile(0.99)), bins=60, color=color, edgecolor="white", alpha=0.8)
    axes[ax_i].set_title(title, fontsize=12)
    axes[ax_i].set_xlabel("交易金额 (amt)")
    axes[ax_i].set_ylabel("频次")
fig.suptitle("交易金额分布：正常 vs 欺诈", fontsize=14, y=1.02)
fig.tight_layout()
p = savefig(fig, "amt_by_fraud.png")
R.append(f"![金额分布对比]({p})\n")

# 5.2 Categorical distributions
R.append("### 5.2 分类特征分布\n")

cat_summary_rows = []
for c in CAT_COLS:
    nuniq = df[c].nunique()
    top_val = df[c].value_counts().index[0] if nuniq > 0 else ""
    top_freq = df[c].value_counts().iloc[0] if nuniq > 0 else 0
    cat_summary_rows.append({
        "特征": c, "唯一值数": nuniq,
        "最高频值": str(top_val)[:40], "最高频次": top_freq,
        "最高频占比(%)": round(top_freq / n_rows * 100, 2),
    })
cat_sum_df = pd.DataFrame(cat_summary_rows)
R.append(md_table(cat_sum_df))
R.append("")

# Category bar chart
print("Generating category distribution …")
fig, ax = plt.subplots(figsize=(12, 5))
cat_counts = df["category"].value_counts()
cat_counts.plot.bar(ax=ax, color="teal", edgecolor="white")
ax.set_title("消费类别 (category) 分布", fontsize=13)
ax.set_ylabel("交易次数")
ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
p = savefig(fig, "category_distribution.png")
R.append(f"![类别分布]({p})\n")

# Gender
fig, ax = plt.subplots(figsize=(6, 4))
df["gender"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%", colors=["#5DA5DA", "#FAA43A"], startangle=90)
ax.set_ylabel("")
ax.set_title("性别分布 (gender)", fontsize=13)
p = savefig(fig, "gender_distribution.png")
R.append(f"![性别分布]({p})\n")

# State top 15
fig, ax = plt.subplots(figsize=(12, 5))
df["state"].value_counts().head(15).plot.bar(ax=ax, color="darkorange", edgecolor="white")
ax.set_title("交易量前 15 的州 (state)", fontsize=13)
ax.set_ylabel("交易次数")
fig.tight_layout()
p = savefig(fig, "state_top15.png")
R.append(f"![州分布]({p})\n")

# 5.3 Time features
R.append("### 5.3 时间特征分布\n")
print("Generating time distributions …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["trans_hour"].value_counts().sort_index().plot.bar(ax=axes[0], color="slateblue", edgecolor="white")
axes[0].set_title("交易时段分布 (小时)", fontsize=12)
axes[0].set_xlabel("小时")
axes[0].set_ylabel("交易次数")
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_counts = df["trans_dayofweek"].value_counts().sort_index()
axes[1].bar(range(7), dow_counts.values, color="mediumseagreen", edgecolor="white")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(day_names)
axes[1].set_title("交易星期分布", fontsize=12)
axes[1].set_ylabel("交易次数")
fig.tight_layout()
p = savefig(fig, "time_distributions.png")
R.append(f"![时间分布]({p})\n")

# ---------------------------------------------------------------------------
# Section 6 — Correlation Analysis
# ---------------------------------------------------------------------------
R.append("## 6. 相关性分析\n")

# Compute correlation matrix for numerical features
corr_cols = [c for c in NUM_COLS if c not in ["zip"]] + [TARGET]
corr_matrix = df[corr_cols].corr()

# 6.1 Heatmap
R.append("### 6.1 数值特征相关性矩阵\n")
print("Generating correlation heatmap …")
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("数值特征相关性矩阵（Pearson）", fontsize=14)
fig.tight_layout()
p = savefig(fig, "correlation_heatmap.png")
R.append(f"![相关性矩阵]({p})\n")

# High correlations
R.append("**高相关性特征对（|r| > 0.5）：**\n")
high_corr = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.5:
            high_corr.append({"特征A": corr_matrix.index[i], "特征B": corr_matrix.columns[j], "相关系数": round(r, 4)})
if high_corr:
    R.append(md_table(pd.DataFrame(high_corr)))
else:
    R.append("未发现 |r| > 0.5 的高相关特征对。")
R.append("")

# 6.2 Hierarchical clustering on correlation
R.append("### 6.2 相关性层次聚类\n")
print("Generating correlation clustering dendrogram …")
try:
    dist_arr = (1 - corr_matrix.abs()).values.copy()
    np.fill_diagonal(dist_arr, 0)
    dist_arr = np.clip(dist_arr, 0, None)
    condensed = squareform(dist_arr, checks=False)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, labels=corr_matrix.columns.tolist(), ax=ax, leaf_rotation=45, leaf_font_size=10)
    ax.set_title("特征相关性层次聚类（Ward 方法）", fontsize=14)
    ax.set_ylabel("距离 (1 - |r|)")
    fig.tight_layout()
    p = savefig(fig, "correlation_dendrogram.png")
    R.append(f"![聚类树状图]({p})\n")
except Exception as e:
    R.append(f"> 聚类树状图生成失败: {e}\n")

# 6.3 Correlation with target (including categorical via point-biserial / cramers V)
R.append("### 6.3 各特征与目标变量 (is_fraud) 的相关性\n")

target_corr_rows = []
for c in NUM_COLS:
    r, p_val = stats.pointbiserialr(df[TARGET], df[c].fillna(0))
    target_corr_rows.append({"特征": c, "类型": "数值", "相关系数(point-biserial r)": round(r, 4), "p-value": f"{p_val:.2e}"})

target_corr_df = pd.DataFrame(target_corr_rows).sort_values("相关系数(point-biserial r)", key=abs, ascending=False)
R.append(md_table(target_corr_df))
R.append("")

print("Generating target correlation bar chart …")
fig, ax = plt.subplots(figsize=(10, 6))
tc = target_corr_df.set_index("特征")["相关系数(point-biserial r)"].sort_values()
colors = ["crimson" if v > 0 else "steelblue" for v in tc.values]
tc.plot.barh(ax=ax, color=colors)
ax.set_title("数值特征与 is_fraud 的 Point-Biserial 相关系数", fontsize=13)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("相关系数")
fig.tight_layout()
p = savefig(fig, "target_correlation.png")
R.append(f"![目标相关性]({p})\n")

# Categorical → target: fraud rate per category
R.append("**分类特征与目标的关系（各类别欺诈率）：**\n")
for cat_col in ["category", "gender"]:
    fraud_rate = df.groupby(cat_col)[TARGET].mean().sort_values(ascending=False)
    R.append(f"\n**{cat_col} 维度欺诈率：**\n")
    fr_df = fraud_rate.reset_index()
    fr_df.columns = [cat_col, "欺诈率"]
    fr_df["欺诈率"] = fr_df["欺诈率"].map(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    R.append(md_table(fr_df))
    R.append("")

# Category fraud rate chart
print("Generating category fraud rate …")
fraud_by_cat = df.groupby("category")[TARGET].mean().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
fraud_by_cat.plot.barh(ax=ax, color="coral", edgecolor="white")
ax.set_title("各消费类别欺诈率", fontsize=13)
ax.set_xlabel("欺诈率")
ax.set_ylabel("")
fig.tight_layout()
p = savefig(fig, "category_fraud_rate.png")
R.append(f"![类别欺诈率]({p})\n")

# Hour fraud rate
print("Generating hourly fraud rate …")
fraud_by_hour = df.groupby("trans_hour")[TARGET].mean()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(fraud_by_hour.index, fraud_by_hour.values, color="tomato", edgecolor="white")
ax.set_title("各时段欺诈率", fontsize=13)
ax.set_xlabel("小时")
ax.set_ylabel("欺诈率")
ax.set_xticks(range(24))
fig.tight_layout()
p = savefig(fig, "hourly_fraud_rate.png")
R.append(f"![时段欺诈率]({p})\n")

# ---------------------------------------------------------------------------
# Section 7 — Target Variable / Class Balance
# ---------------------------------------------------------------------------
R.append("## 7. 目标变量与样本平衡分析\n")

fraud_counts = df[TARGET].value_counts()
n_normal = fraud_counts.get(0, 0)
n_fraud = fraud_counts.get(1, 0)
imbalance_ratio = n_normal / max(n_fraud, 1)

R.append(f"| 类别 | 样本数 | 占比 |")
R.append(f"|---|---|---|")
R.append(f"| 正常 (0) | {n_normal:,} | {n_normal/n_rows*100:.2f}% |")
R.append(f"| 欺诈 (1) | {n_fraud:,} | {n_fraud/n_rows*100:.2f}% |")
R.append(f"| **不平衡比** | **{imbalance_ratio:.1f} : 1** | |")
R.append("")

R.append(f"- 正常样本数: **{n_normal:,}**")
R.append(f"- 欺诈样本数: **{n_fraud:,}**")
R.append(f"- 不平衡比: **{imbalance_ratio:.1f} : 1**（严重不平衡）")
R.append(f"- 目标变量缺失: **{df[TARGET].isnull().sum()}**\n")

print("Generating class balance chart …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fraud_counts.plot.bar(ax=axes[0], color=["steelblue", "crimson"], edgecolor="white")
axes[0].set_title("样本数量", fontsize=12)
axes[0].set_xticklabels(["正常 (0)", "欺诈 (1)"], rotation=0)
axes[0].set_ylabel("样本数")
for i, v in enumerate(fraud_counts.values):
    axes[0].text(i, v + n_rows * 0.01, f"{v:,}", ha="center", fontsize=10)

fraud_counts.plot.pie(ax=axes[1], autopct="%1.2f%%", colors=["steelblue", "crimson"],
                      labels=["正常 (0)", "欺诈 (1)"], startangle=90)
axes[1].set_title("样本占比", fontsize=12)
axes[1].set_ylabel("")
fig.suptitle("目标变量 (is_fraud) 类别分布", fontsize=14, y=1.02)
fig.tight_layout()
p = savefig(fig, "class_balance.png")
R.append(f"![类别平衡]({p})\n")

# ---------------------------------------------------------------------------
# Section 8 — Summary Statistics per Target
# ---------------------------------------------------------------------------
R.append("## 8. 正常 vs 欺诈样本的数值特征对比\n")

compare_cols = ["amt", "age", "city_pop", "lat", "long", "merch_lat", "merch_long"]
compare_cols = [c for c in compare_cols if c in df.columns]
comp_stats = df.groupby(TARGET)[compare_cols].agg(["mean", "median", "std"]).round(4)
comp_stats.columns = [f"{c}_{s}" for c, s in comp_stats.columns]
comp_stats = comp_stats.T
comp_stats.columns = ["正常 (0)", "欺诈 (1)"]
comp_stats.index.name = "特征_统计量"
R.append(md_table(comp_stats.reset_index()))
R.append("")

# ---------------------------------------------------------------------------
# Section 9 — Recommendations
# ---------------------------------------------------------------------------
R.append("## 9. 建模前建议\n")
R.append(dedent("""\
### 9.1 数据质量
- 数据集无缺失值，数据完整性良好。
- `amt`（交易金额）和 `city_pop`（城市人口）存在较多异常值（右偏分布），建模时可考虑对数变换或 RobustScaler。

### 9.2 类别不平衡处理
- 欺诈样本仅占约 0.58%，属于严重不平衡。
- 建议策略：SMOTE 过采样、欠采样、调整 `class_weight`、或使用 Focal Loss。
- 评估指标应以 **Precision、Recall、F1-Score、AUC-ROC、AUC-PR** 为主，不应使用 Accuracy。

### 9.3 特征工程建议
- **时间特征**: 交易小时、星期几（已提取），可进一步提取月份、是否节假日等。
- **地理特征**: 计算持卡人位置与商户位置的距离（`lat/long` vs `merch_lat/merch_long`）。
- **金额特征**: 对 `amt` 取对数变换；计算用户历史平均交易额的偏差。
- **类别编码**: `category` 可用 Target Encoding 或 Label Encoding；`merchant`、`job` 等高基数特征可考虑 Frequency Encoding。
- **ID 类字段**: `row_index`、`trans_num`、`cc_num`、`unix_time` 不建议直接作为特征。
- **个人信息**: `first`、`last`、`street` 等对欺诈检测意义不大，建议剔除。

### 9.4 推荐模型
- **基线模型**: Logistic Regression
- **树模型**: LightGBM / XGBoost / Random Forest（通常在此类任务中表现最优）
- **深度学习**: 可尝试但表格数据上通常不优于树模型
- **集成方法**: Stacking / Blending 多个模型
"""))

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
print("Writing report …")
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n✅ EDA report saved to: {REPORT_PATH}")
print(f"   Figures saved to:    {FIG_DIR}/")
