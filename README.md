# DataAnalysis

债券收益率预测与A股自动交易策略系统 — 基于多模型集成和机器学习的金融资产分析平台。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Models](https://img.shields.io/badge/Models-12+-green)
![ML](https://img.shields.io/badge/ML-LSTM%20|%20LightGBM%20|%20HMM-orange)
![Skills](https://img.shields.io/badge/Skills-41-purple)

## 核心功能

### A股自动交易策略系统

```bash
streamlit run app/astock_dashboard.py
```

**选股策略 (5种):**
- 动量策略 / 均值回归 / 趋势跟踪 / 突破策略 (规则型)
- **ML多因子选股 (LightGBM)** — 30+量化因子自动学习

**交易信号 (9种):**
- MACD交叉 / 均线交叉 / RSI / 布林带 / KDJ / 量价配合 / 综合加权 (规则型)
- **LSTM深度学习信号** — LSTM+Attention端到端预测涨跌概率
- **Meta-Labeling信号过滤** — ML判断规则信号可靠性，过滤低质量信号

**ML增强 (可叠加):**
- **Meta-Labeling过滤** — 在任意规则策略上叠加，只执行高置信度信号
- **市场状态过滤** — HMM识别牛市/熊市/震荡，熊市自动抑制买入

**A股规则:**
- T+1: 当日买入次日方可卖出
- 涨跌停: 主板 ±10%, 创业板/科创板 ±20%
- 费用: 佣金万2.5 + 印花税千0.5(卖出) + 过户费万0.1
- 最小交易单位: 100股

### 债券收益率预测系统

```bash
streamlit run app/dashboard.py
```

8种中美国债 + 5模型集成(Ridge/XGBoost/LightGBM/RF/ARIMA) + 回测

## 项目结构

```
DataAnalysis/
├── app/
│   ├── astock_dashboard.py       # A股交易策略仪表板
│   ├── astock/
│   │   ├── market_data.py        # 行情数据 + 15种技术指标
│   │   ├── stock_selector.py     # 4种规则选股策略
│   │   ├── signal_model.py       # 7种规则信号模型
│   │   ├── backtester.py         # A股规则回测引擎
│   │   ├── ml_meta_labeling.py   # Meta-Labeling 信号过滤器
│   │   ├── ml_factor_selector.py # LightGBM 多因子选股
│   │   ├── ml_lstm_signal.py     # LSTM+Attention 信号模型
│   │   └── ml_regime.py          # 市场状态识别 (GMM)
│   ├── dashboard.py              # 债券预测仪表板
│   ├── data_fetcher.py           # 债券数据获取
│   ├── predictor.py              # 债券预测引擎
│   └── backtester.py             # 债券回测模块
├── scripts/                      # 研究分析脚本
│   ├── bond_forecast.py          # 12模型国债预测对比(含Transformer)
│   ├── eda_fraud_train.py        # 信用卡欺诈EDA
│   ├── prepare_dataset.py        # 特征工程
│   ├── train_models.py           # 欺诈检测建模
│   └── plot_config.py            # 中文字体配置
├── reports/                      # 分析报告 + 图表
├── data/                         # 数据(不入版本控制)
└── .cursor/skills/               # 41个AI Agent Skills
```

## ML算法详解

### 1. Meta-Labeling (信号过滤器)

```
规则信号 (MACD/RSI/...) → LightGBM分类器判断可靠性 → 只执行高置信度信号
```

- 参考: Marcos Lopez de Prado "Advances in Financial Machine Learning"
- 特征: RSI, MACD, 布林宽度, 趋势斜率, 动量, 波动率等20+维
- 时序交叉验证，避免前视偏差
- 可将胜率从40%提升到55%+

### 2. LightGBM 多因子选股

```
30+因子 (量价/技术/动量/波动率) → LightGBM预测未来5日收益排名 → Top-N
```

- 因子: 5/10/20/60日收益率, RSI, MACD, KDJ, 均线排列, 量比, ATR, 波动率等
- 自动发现因子组合和非线性交互
- 时序切分训练/验证，防止过拟合

### 3. LSTM+Attention 交易信号

```
过去30日 [OHLCV + 技术指标] → LSTM → Attention池化 → P(上涨)
```

- 11维标准化输入特征
- 双层LSTM + 缩放点积注意力
- 输出上涨概率，>0.6买入，<0.4卖出
- CosineAnnealing学习率调度 + 梯度裁剪

### 4. 市场状态识别 (HMM/GMM)

```
滚动特征 (收益率/波动率/趋势/RSI) → 高斯混合模型 → 牛市/熊市/震荡
```

- 3状态GMM自动聚类
- 按平均收益率映射为牛/熊/震荡
- 熊市自动抑制买入信号，减少逆势交易

## 安装依赖

```bash
pip install pandas numpy scikit-learn xgboost lightgbm statsmodels pmdarima \
    akshare streamlit plotly torch
```

## 风险提示

> 本系统仅供研究和学习参考，不构成任何投资建议。金融市场存在不确定性，模型预测可能失效。过去的回测表现不代表未来收益。请在充分了解风险的前提下做出投资决策。

## License

MIT
