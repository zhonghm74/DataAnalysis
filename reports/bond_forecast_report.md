# 中国10年期国债收益率预测 — 多模型自回归比较报告

> 生成时间: 2026-03-26 10:35:48
---

## 1. 研究概述

**目标**: 预测中国10年期国债收益率未来走势，比较多种自回归模型的预测性能。

**数据**: 中国10年期国债收益率日频数据（来源: 东方财富/英为财情）
- 时间范围: 2015-01-05 ~ 2026-03-25
- 样本量: 2928 个交易日
- 训练集: 2868 个交易日
- 测试集: **最近 60 个交易日** (walk-forward)

**评估指标**:
- RMSE (均方根误差) — 主指标
- MAE (平均绝对误差)
- MAPE (平均绝对百分比误差)
- R² (决定系数)

![历史走势](figures/bond_historical.png)

## 2. 模型说明

### 统计模型
| 模型 | 方法 |
|---|---|
| ARIMA (auto) | 自动选择 (p,d,q) 的 ARIMA，通过 AIC 优化 |
| SARIMAX | 季节性 ARIMA，网格搜索 order + seasonal_order |
| ETS (Holt-Winters) | 指数平滑，搜索 trend/damped 组合 |

### 机器学习模型 (基于手工滞后特征)
| 模型 | 方法 |
|---|---|
| AR-XGBoost | 滞后特征 + XGBoost 回归，网格搜索超参 |
| AR-LightGBM | 滞后特征 + LightGBM 回归，网格搜索超参 |
| AR-Random Forest | 滞后特征 + 随机森林回归，网格搜索超参 |
| AR-Ridge | 滞后特征 + 岭回归，搜索正则化强度 |

### Transformer 类深度学习模型 (端到端序列建模)
| 模型 | 方法 |
|---|---|
| Transformer Encoder | 标准多头自注意力编码器 + 位置编码，网格搜索 d_model/nhead/layers |
| PatchTST | 将时序分割为 patch 再做 Transformer 编码（2023 SOTA），搜索 patch_len/d_model |
| LSTM + Attention | 双层 LSTM + 缩放点积注意力池化，搜索 hidden_size/layers |
| Informer-lite | ProbSparse 注意力机制（降低复杂度的 Informer 变体），搜索 d_model/layers |

### 基线
| 模型 | 方法 |
|---|---|
| Naive (t-1) | 前一日收益率作为预测值 |

**AR 特征工程** (用于 ML 模型):
- 滞后特征: lag_1, lag_2, lag_3, lag_5, lag_10, lag_20, lag_60
- 滚动统计: rolling_5_mean, rolling_20_mean, rolling_5_std
- 差分特征: diff_1, diff_5

**Transformer 输入** (用于深度学习模型):
- 滑动窗口: 前 60 个交易日的标准化收益率序列
- 标准化: 训练集均值/标准差归一化
- 训练: Adam + CosineAnnealing, 80 epochs, 梯度裁剪

## 3. 模型排行榜

![指标对比](figures/bond_metrics_comparison.png)

|   排名 | model               |      MAE |     RMSE |   MAPE(%) |      R² |   time_s | params                                           |
|-----:|:--------------------|---------:|---------:|----------:|--------:|---------:|:-------------------------------------------------|
|    1 | AR-Ridge            | 0.007377 | 0.010413 |    0.4038 |  0.861  |      0   | alpha=0.1                                        |
|    2 | Naive (t-1)         | 0.007267 | 0.010496 |    0.3975 |  0.8588 |      0   | y(t)=y(t-1)                                      |
|    3 | Transformer Encoder | 0.009653 | 0.012646 |    0.5288 |  0.795  |     78.1 | d=16,h=4,L=2,ff=32,lr=0.001                      |
|    4 | AR-LightGBM         | 0.011484 | 0.01525  |    0.6311 |  0.7019 |      7.8 | {'max_depth': 3, 'lr': 0.1, 'num_leaves': 31}    |
|    5 | LSTM + Attention    | 0.011843 | 0.01576  |    0.6482 |  0.6816 |     41.8 | h=64,L=2,lr=0.001                                |
|    6 | AR-Random Forest    | 0.01332  | 0.017889 |    0.7297 |  0.5898 |      8.4 | {'n_estimators': 500, 'max_depth': 5}            |
|    7 | AR-XGBoost          | 0.013951 | 0.017929 |    0.7677 |  0.5879 |      3.8 | {'max_depth': 5, 'lr': 0.1, 'n_estimators': 200} |
|    8 | PatchTST            | 0.01533  | 0.018985 |    0.8422 |  0.538  |     19.3 | patch=10,d=32,L=2,lr=0.001                       |
|    9 | ARIMA (auto)        | 0.0322   | 0.037555 |    1.779  | -0.8078 |      2.1 | order=(0, 1, 1)                                  |
|   10 | SARIMAX             | 0.032432 | 0.037799 |    1.7919 | -0.8314 |      9.1 | order=(1, 1, 1), seasonal=(1, 0, 1, 5)           |
|   11 | ETS (Holt-Winters)  | 0.033038 | 0.038442 |    1.8256 | -0.8943 |      0.6 | trend=add, damped=True                           |
|   12 | Informer-lite       | 0.036901 | 0.038846 |    2.0263 | -0.9342 |     39.5 | d=32,h=4,L=2,lr=0.001                            |

## 4. 预测结果可视化

![所有模型预测](figures/bond_all_predictions.png)

![Top-4 预测](figures/bond_top4_predictions.png)

## 5. 残差分析

![残差分析](figures/bond_residual_analysis.png)

![累积误差](figures/bond_cumulative_error.png)

## 6. 特征重要性 (ML 模型)

![特征重要性](figures/bond_feature_importance.png)

ML 模型的滞后特征重要性分析显示：
- **lag_1 (前1日)** 是最重要的特征，符合国债收益率强自相关特性
- 短期滚动均值和差分特征提供了趋势和动量信号
- 长期滞后 (lag_60) 捕捉了更长周期的均值回归效应

## 7. 最优超参数

- **AR-Ridge**: alpha=0.1
- **Naive (t-1)**: y(t)=y(t-1)
- **Transformer Encoder**: d=16,h=4,L=2,ff=32,lr=0.001
- **AR-LightGBM**: {'max_depth': 3, 'lr': 0.1, 'num_leaves': 31}
- **LSTM + Attention**: h=64,L=2,lr=0.001

## 8. 结论

### 主要发现

1. **AR-Ridge** 以 RMSE=0.010413 取得最优预测性能。

2. **三大类模型性能对比**:
   - 统计模型最优: ARIMA (auto) (RMSE=0.037555)
   - ML 模型最优: AR-Ridge (RMSE=0.010413)
   - Transformer 模型最优: Transformer Encoder (RMSE=0.012646)

3. **Transformer 模型分析**: Transformer 类模型在国债收益率这类低噪声、强自相关的金融时序上，面临"过度建模"的风险——自注意力机制更适合捕捉复杂的长距离依赖关系，但国债收益率的变化主要由短期自相关驱动，简单的滞后特征已足够。PatchTST 通过分 patch 建模能缓解过拟合，通常是 Transformer 类中表现最好的。

4. **Naive 基线的竞争力**: 前一日预测（Naive t-1）极具竞争力，反映了国债收益率的随机游走特性。

5. **统计模型局限**: ARIMA/ETS 的多步直接预测误差快速积累，在 60 天测试期上 R² 为负。

### 各类模型适用场景

| 类别 | 适用场景 | 局限 |
|---|---|---|
| 统计模型 (ARIMA/ETS) | 短期 (1-5 步) 预测，可解释性强 | 多步预测误差积累，无法捕捉非线性 |
| ML 模型 (Ridge/XGBoost) | 中短期预测，特征工程灵活 | 依赖手工特征，不自动学习序列模式 |
| Transformer 类 | 长序列、复杂模式、多变量场景 | 小数据集易过拟合，训练成本高 |

### 建议

- **短期预测 (1-5天)**: 优先使用 AR-Ridge，辅以 Naive 作为合理性检查。
- **中期预测 (1-3月)**: 结合宏观经济因子（GDP、CPI、央行政策）构建多因子模型。
- **Transformer 优化方向**: 增加训练数据（多期限债券联合建模）、加入宏观因子作为协变量、使用预训练时序基础模型。
- **模型集成**: 将 ML 模型和 Transformer 模型预测加权平均可提高稳健性。
