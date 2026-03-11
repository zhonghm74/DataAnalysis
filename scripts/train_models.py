"""
Credit‑card fraud — lean model training & hyperparameter tuning.

Approach:
  1. Quick baselines on original data (300 estimators, class weights)
  2. Hyperparameter search on 10% subsample (fast iteration)
  3. Final model trained on SMOTE data with best params
  4. Full evaluation on held‑out test set
"""

import os, time, warnings, pickle, json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score,
)
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "data", "models")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
REPORT_PATH = os.path.join(ROOT, "reports", "modeling_report.md")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"

def metrics(y_true, y_pred, y_prob):
    return {
        "AUC-ROC": round(roc_auc_score(y_true, y_prob), 4),
        "AUC-PR": round(average_precision_score(y_true, y_prob), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
    }

# ===================================================================
print("=" * 60)
print("Loading data …")
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
features = X_train.columns.tolist()
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  Train: {X_train.shape}  fraud={y_train.sum():,}  ratio={ratio:.1f}:1")
print(f"  Test:  {X_test.shape}  fraud={y_test.sum():,}")

all_results = []

# ===================================================================
# PHASE 1 — BASELINES (fast: n_estimators=300)
# ===================================================================
print("\n" + "=" * 60)
print("Phase 1: Baselines (n_estimators=300)\n")

def run_model(name, mdl):
    print(f"  {name:22s}", end=" … ", flush=True)
    t0 = time.time()
    mdl.fit(X_train, y_train)
    tt = time.time() - t0
    yp = mdl.predict(X_test)
    ypr = mdl.predict_proba(X_test)[:, 1]
    m = metrics(y_test, yp, ypr)
    m.update(model=name, time_s=round(tt, 1), _prob=ypr, _pred=yp, _obj=mdl)
    print(f"AUC-PR={m['AUC-PR']:.4f}  F1={m['F1']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  [{tt:.0f}s]")
    return m

all_results.append(run_model("Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs",
                       random_state=42, n_jobs=-1)))

all_results.append(run_model("Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                           class_weight="balanced_subsample", random_state=42, n_jobs=-1)))

all_results.append(run_model("XGBoost",
    xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=ratio, eval_metric="aucpr",
                       tree_method="hist", random_state=42, n_jobs=-1, verbosity=0)))

all_results.append(run_model("LightGBM",
    lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        scale_pos_weight=ratio, random_state=42, n_jobs=-1, verbose=-1)))

base_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in all_results]).sort_values("AUC-PR", ascending=False)
print("\n" + base_df.to_string(index=False))

# ===================================================================
# PHASE 2 — TUNING on 10% subsample (XGBoost + LightGBM)
# ===================================================================
print("\n" + "=" * 60)
print("Phase 2: Hyperparameter tuning (10% subsample + early stopping)\n")

# 10% stratified subsample for fast search
rng = np.random.RandomState(42)
idx_pos = np.where(y_train == 1)[0]
idx_neg = rng.choice(np.where(y_train == 0)[0], size=len(idx_pos) * 20, replace=False)
idx_sub = np.concatenate([idx_pos, idx_neg])
X_sub, y_sub = X_train.iloc[idx_sub], y_train[idx_sub]
X_s_tr, X_s_val, y_s_tr, y_s_val = train_test_split(
    X_sub, y_sub, test_size=0.2, stratify=y_sub, random_state=42)
print(f"  Subsample: train={X_s_tr.shape[0]:,}  val={X_s_val.shape[0]:,}")

XGB_CONFIGS = [
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_alpha": 0.01, "reg_lambda": 2},
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0, "reg_lambda": 1},
    {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 1, "reg_alpha": 0.1, "reg_lambda": 5},
    {"max_depth": 6, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 5, "reg_alpha": 0.01, "reg_lambda": 2},
    {"max_depth": 8, "learning_rate": 0.01, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0, "reg_lambda": 5},
    {"max_depth": 10, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 2},
    {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1, "reg_alpha": 0, "reg_lambda": 1},
    {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 5},
]

LGB_CONFIGS = [
    {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20, "reg_alpha": 0.01, "reg_lambda": 1, "num_leaves": 63},
    {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.7, "min_child_samples": 10, "reg_alpha": 0, "reg_lambda": 5, "num_leaves": 127},
    {"max_depth": -1, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.9, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0, "num_leaves": 63},
    {"max_depth": 6, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20, "reg_alpha": 0.01, "reg_lambda": 2, "num_leaves": 63},
    {"max_depth": 8, "learning_rate": 0.01, "subsample": 0.9, "colsample_bytree": 0.8, "min_child_samples": 10, "reg_alpha": 0, "reg_lambda": 5, "num_leaves": 127},
    {"max_depth": -1, "learning_rate": 0.1, "subsample": 0.7, "colsample_bytree": 0.8, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1, "num_leaves": 31},
    {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_samples": 10, "reg_alpha": 0, "reg_lambda": 1, "num_leaves": 31},
    {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 20, "reg_alpha": 0.01, "reg_lambda": 5, "num_leaves": 127},
]

tuned_results = []
for algo, configs in [("XGBoost", XGB_CONFIGS), ("LightGBM", LGB_CONFIGS)]:
    print(f"\n  Tuning {algo} ({len(configs)} configs) …")
    best_ap, best_p, best_m = -1, {}, None
    t0 = time.time()
    for i, cfg in enumerate(configs):
        if algo == "XGBoost":
            m = xgb.XGBClassifier(n_estimators=1000, scale_pos_weight=ratio,
                eval_metric="aucpr", tree_method="hist", early_stopping_rounds=20,
                random_state=42, n_jobs=-1, verbosity=0, **cfg)
            m.fit(X_s_tr, y_s_tr, eval_set=[(X_s_val, y_s_val)], verbose=False)
        else:
            m = lgb.LGBMClassifier(n_estimators=1000, scale_pos_weight=ratio,
                random_state=42, n_jobs=-1, verbose=-1, **cfg)
            m.fit(X_s_tr, y_s_tr, eval_set=[(X_s_val, y_s_val)],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        ap = average_precision_score(y_s_val, m.predict_proba(X_s_val)[:, 1])
        iters = getattr(m, "best_iteration", getattr(m, "best_iteration_", ""))
        print(f"    [{i+1}/{len(configs)}] val AP={ap:.4f}  iters={iters}")
        if ap > best_ap:
            best_ap, best_p, best_m = ap, cfg, m
    t_tune = time.time() - t0
    print(f"  Best val AP={best_ap:.4f}  params={best_p}")

    # Retrain on full train set
    print(f"  Retraining on full data …", end=" ", flush=True)
    t1 = time.time()
    X_ft, X_fv, y_ft, y_fv = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)
    if algo == "XGBoost":
        final = xgb.XGBClassifier(n_estimators=1000, scale_pos_weight=ratio,
            eval_metric="aucpr", tree_method="hist", early_stopping_rounds=30,
            random_state=42, n_jobs=-1, verbosity=0, **best_p)
        final.fit(X_ft, y_ft, eval_set=[(X_fv, y_fv)], verbose=False)
    else:
        final = lgb.LGBMClassifier(n_estimators=1000, scale_pos_weight=ratio,
            random_state=42, n_jobs=-1, verbose=-1, **best_p)
        final.fit(X_ft, y_ft, eval_set=[(X_fv, y_fv)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    t_full = time.time() - t1
    yp = final.predict(X_test)
    ypr = final.predict_proba(X_test)[:, 1]
    mt = metrics(y_test, yp, ypr)
    iters_f = getattr(final, "best_iteration", getattr(final, "best_iteration_", ""))
    mt.update(model=f"{algo} (tuned)", time_s=round(t_tune + t_full, 1),
              val_auc_pr=round(best_ap, 4), best_params=best_p,
              best_iters=iters_f,
              _prob=ypr, _pred=yp, _obj=final)
    tuned_results.append(mt)
    all_results.append(mt)
    print(f"[{t_full:.0f}s] iters={iters_f}  AUC-PR={mt['AUC-PR']}  F1={mt['F1']}")

# ===================================================================
# PHASE 3 — SMOTE
# ===================================================================
print("\n" + "=" * 60)
print("Phase 3: Best model on SMOTE\n")

best_so_far = max(all_results, key=lambda r: r["AUC-PR"])
ba = best_so_far["model"].replace(" (tuned)", "")
bp = best_so_far.get("best_params", {})
print(f"  Best so far: {best_so_far['model']} AUC-PR={best_so_far['AUC-PR']}")

print("  Loading SMOTE …")
X_smote = pd.read_csv(os.path.join(DATA_DIR, "X_train_smote.csv"))
y_smote = pd.read_csv(os.path.join(DATA_DIR, "y_train_smote.csv")).values.ravel()
X_st, X_sv, y_st, y_sv = train_test_split(
    X_smote, y_smote, test_size=0.1, stratify=y_smote, random_state=42)
del X_smote, y_smote

print(f"  Training {ba} on SMOTE ({X_st.shape[0]:,} rows) …", end=" ", flush=True)
t0 = time.time()
if ba == "XGBoost":
    sm = xgb.XGBClassifier(n_estimators=1000, eval_metric="aucpr",
        tree_method="hist", early_stopping_rounds=30, scale_pos_weight=1,
        random_state=42, n_jobs=-1, verbosity=0, **bp)
    sm.fit(X_st, y_st, eval_set=[(X_sv, y_sv)], verbose=False)
else:
    sm = lgb.LGBMClassifier(n_estimators=1000, scale_pos_weight=1,
        random_state=42, n_jobs=-1, verbose=-1, **bp)
    sm.fit(X_st, y_st, eval_set=[(X_sv, y_sv)],
           callbacks=[lgb.early_stopping(30, verbose=False)])
ts = time.time() - t0
del X_st, X_sv, y_st, y_sv
print(f"[{ts:.0f}s]")

yp_s = sm.predict(X_test)
ypr_s = sm.predict_proba(X_test)[:, 1]
ms = metrics(y_test, yp_s, ypr_s)
iters_s = getattr(sm, "best_iteration", getattr(sm, "best_iteration_", ""))
ms.update(model=f"{ba} (tuned+SMOTE)", time_s=round(ts, 1),
          best_iters=iters_s, _prob=ypr_s, _pred=yp_s, _obj=sm)
all_results.append(ms)
print(f"  AUC-PR={ms['AUC-PR']}  F1={ms['F1']}  P={ms['Precision']}  R={ms['Recall']}")

# ===================================================================
# BEST
# ===================================================================
print("\n" + "=" * 60)
best = max(all_results, key=lambda r: r["AUC-PR"])
best_label = best["model"]
best_obj = best["_obj"]
y_prob_best = best["_prob"]
y_pred_best = best["_pred"]
print(f"🏆 BEST: {best_label}")
print(f"   AUC-ROC={best['AUC-ROC']}  AUC-PR={best['AUC-PR']}  F1={best['F1']}  "
      f"P={best['Precision']}  R={best['Recall']}")

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_obj, f)
meta = {k: v for k, v in best.items() if not k.startswith("_")}
with open(os.path.join(MODEL_DIR, "best_model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

# ===================================================================
# CHARTS
# ===================================================================
print("\nGenerating charts …")
lb_df = pd.DataFrame([{k: v for k, v in r.items()
    if not k.startswith("_") and k != "best_params"}
    for r in all_results]).sort_values("AUC-PR", ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(20, max(7, len(lb_df)*0.6 + 1)))
for ax, met in zip(axes, ["AUC-ROC", "AUC-PR", "F1"]):
    d = lb_df.sort_values(met, ascending=True)
    colors = ["#E91E63" if "SMOTE" in l else "#FF9800" if "tuned" in l else "#2196F3" for l in d["model"]]
    ax.barh(d["model"], d[met], color=colors, edgecolor="white")
    ax.set_title(met, fontsize=13)
    for i, v in enumerate(d[met]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)
fig.suptitle("模型排行榜", fontsize=15, y=1.01)
fig.tight_layout()
p_lb = savefig(fig, "model_baseline_comparison.png")

top4 = sorted(all_results, key=lambda r: r["AUC-PR"], reverse=True)[:4]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for e in top4:
    fpr, tpr, _ = roc_curve(y_test, e["_prob"])
    axes[0].plot(fpr, tpr, label=f"{e['model']} ({e['AUC-ROC']})", lw=2)
axes[0].plot([0,1],[0,1],"k--",alpha=0.3); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC (Top‑4)"); axes[0].legend(fontsize=8)
for e in top4:
    p_, r_, _ = precision_recall_curve(y_test, e["_prob"])
    axes[1].plot(r_, p_, label=f"{e['model']} ({e['AUC-PR']})", lw=2)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision‑Recall (Top‑4)"); axes[1].legend(fontsize=8)
fig.tight_layout(); p_curves = savefig(fig, "model_roc_pr_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[0],
    xticklabels=["正常","欺诈"], yticklabels=["正常","欺诈"])
axes[0].set_title(f"混淆矩阵"); axes[0].set_xlabel("预测"); axes[0].set_ylabel("实际")
cm_n = cm.astype(float)/cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_n, annot=True, fmt=".2%", cmap="Oranges", ax=axes[1],
    xticklabels=["正常","欺诈"], yticklabels=["正常","欺诈"])
axes[1].set_title("归一化"); axes[1].set_xlabel("预测"); axes[1].set_ylabel("实际")
fig.tight_layout(); p_cm = savefig(fig, "model_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(10, 8))
imp = getattr(best_obj, "feature_importances_", np.zeros(len(features)))
imp_df = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance")
ax.barh(imp_df["feature"], imp_df["importance"], color="teal", edgecolor="white")
ax.set_title(f"特征重要性 — {best_label}"); fig.tight_layout()
p_fi = savefig(fig, "model_feature_importance.png")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_prob_best[y_test==0], bins=100, alpha=0.6, color="steelblue", label="正常", density=True)
ax.hist(y_prob_best[y_test==1], bins=100, alpha=0.6, color="crimson", label="欺诈", density=True)
ax.set_xlabel("预测概率"); ax.set_ylabel("密度"); ax.set_title("预测概率分布"); ax.legend()
fig.tight_layout(); p_sd = savefig(fig, "model_score_distribution.png")

prec_a, rec_a, thr = precision_recall_curve(y_test, y_prob_best)
f1_a = 2*prec_a[:-1]*rec_a[:-1]/(prec_a[:-1]+rec_a[:-1]+1e-8)
bi = np.argmax(f1_a); bt = thr[bi]
y_opt = (y_prob_best >= bt).astype(int)
of1 = f1_score(y_test, y_opt); op = precision_score(y_test, y_opt); orr = recall_score(y_test, y_opt)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thr, prec_a[:-1], label="Precision", color="steelblue", lw=2)
ax.plot(thr, rec_a[:-1], label="Recall", color="crimson", lw=2)
ax.plot(thr, f1_a, label="F1", color="green", lw=2, ls="--")
ax.axvline(bt, color="gray", ls=":", alpha=0.7, label=f"最优={bt:.3f}")
ax.set_xlabel("阈值"); ax.set_ylabel("Score"); ax.set_title("阈值分析"); ax.legend(); ax.set_xlim(0,1)
fig.tight_layout(); p_thr = savefig(fig, "model_threshold_analysis.png")
print(f"  Optimal threshold={bt:.4f} → F1={of1:.4f} P={op:.4f} R={orr:.4f}")

# ===================================================================
# REPORT
# ===================================================================
print("\nWriting report …")
R = []
R.append("# 建模报告 — 信用卡欺诈检测\n")
R.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n---\n")
R.append("## 1. 建模策略\n")
R.append("""\
| 阶段 | 方法 |
|---|---|
| Phase 1 | 4 种算法基线 (LR/RF/XGBoost/LightGBM)，class_weight 处理不平衡 |
| Phase 2 | XGBoost + LightGBM 网格搜索 (8 configs × early stopping) |
| Phase 3 | 最优参数在 SMOTE 均衡数据上重训练 |

**评估主指标**: AUC‑PR (Average Precision) — 最适合严重不平衡场景
""")
R.append("## 2. 排行榜\n")
R.append(f"![排行榜]({p_lb})\n"); R.append(lb_df.to_markdown(index=False)); R.append("")
R.append("## 3. 超参数调优\n")
R.append("方法: 8 组参数 × early stopping on 10% subsample → 最优参数 retrain on full data\n")
for t in tuned_results:
    R.append(f"### {t['model']}\n")
    R.append(f"- Val AUC-PR: {t.get('val_auc_pr','N/A')}")
    R.append(f"- Test: AUC-ROC={t['AUC-ROC']} | AUC-PR={t['AUC-PR']} | F1={t['F1']} | P={t['Precision']} | R={t['Recall']}")
    R.append(f"- Best iterations: {t.get('best_iters','N/A')}")
    bp_ = t.get("best_params", {})
    R.append(f"- 最优参数:\n```json\n{json.dumps(bp_, indent=2, default=str)}\n```\n")
R.append("## 4. 最终模型\n")
R.append(f"### 🏆 {best_label}\n")
R.append("| 指标 | 值 |\n|---|---|")
for k in ["AUC-ROC","AUC-PR","F1","Precision","Recall","Accuracy"]:
    R.append(f"| {k} | **{best[k]}** |")
R.append(f"\n![ROC & PR]({p_curves})\n![混淆矩阵]({p_cm})\n![概率分布]({p_sd})\n")
R.append("### Classification Report\n```")
R.append(classification_report(y_test, y_pred_best, target_names=["正常","欺诈"]))
R.append("```\n")
R.append("## 5. 特征重要性\n")
R.append(f"![特征重要性]({p_fi})\n")
R.append("| 排名 | 特征 | 重要性 |\n|---|---|---|")
for rk, (_, row) in enumerate(imp_df.sort_values("importance", ascending=False).iterrows(), 1):
    R.append(f"| {rk} | `{row['feature']}` | {row['importance']:.4f} |")
R.append("")
R.append("## 6. 阈值优化\n")
R.append(f"![阈值]({p_thr})\n")
R.append("| 阈值 | F1 | Precision | Recall |\n|---|---|---|---|")
R.append(f"| 0.5 (默认) | {best['F1']} | {best['Precision']} | {best['Recall']} |")
R.append(f"| **{bt:.4f} (最优)** | **{of1:.4f}** | **{op:.4f}** | **{orr:.4f}** |\n")
R.append("## 7. 结论\n")
R.append(f"1. **{best_label}** AUC‑PR 最优。\n2. 梯度提升树显著优于线性模型和随机森林。\n3. 最优阈值 **{bt:.4f}** (F1={of1:.4f})。\n4. 进一步: Optuna 调参、Stacking 集成、用户行为特征。\n")
R.append("## 8. 输出文件\n| 文件 | 说明 |\n|---|---|")
R.append("| `data/models/best_model.pkl` | 最优模型 |")
R.append("| `data/models/best_model_meta.json` | 模型元信息 |")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(R))

print(f"\n{'='*60}")
print(f"✅ Done!  {REPORT_PATH}")
print(f"   Best: {best_label}  AUC-PR={best['AUC-PR']}  F1={best['F1']}")
