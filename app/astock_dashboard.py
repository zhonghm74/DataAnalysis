"""
A-Share Automated Trading Strategy Dashboard.

Features:
  - Stock selection strategies (momentum, mean-reversion, trend, breakout)
  - 7 signal models (MACD, MA cross, RSI, Bollinger, KDJ, Volume-Price, Composite)
  - Full backtesting with A-share rules (T+1, fees, price limits)
  - Interactive Plotly charts
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from astock.market_data import fetch_stock, add_technical_indicators, STOCK_POOLS
from astock.stock_selector import SELECTORS, ML_SELECTORS
from astock.signal_model import SIGNAL_MODELS, ML_SIGNAL_MODELS
from astock.backtester import run_backtest, BacktestConfig
from astock.ml_meta_labeling import MetaLabeler
from astock.ml_factor_selector import MLFactorSelector
from astock.ml_lstm_signal import LSTMSignalModel
from astock.ml_regime import MarketRegimeDetector

st.set_page_config(page_title="A股自动交易策略系统", page_icon="🇨🇳", layout="wide")

st.markdown("""
<style>
    .signal-buy {background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 15px; border-radius: 10px; color: white; text-align: center;}
    .signal-sell {background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 15px; border-radius: 10px; color: white; text-align: center;}
    .signal-hold {background: linear-gradient(135deg, #4facfe, #00f2fe);
        padding: 15px; border-radius: 10px; color: white; text-align: center;}
    .metric-box {background: #f8f9fa; padding: 12px; border-radius: 8px;
        border-left: 4px solid #2196F3; margin: 4px 0;}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.title("🇨🇳 A股交易策略")

    page = st.radio("功能模块", ["📊 选股策略", "📈 交易信号 & 回测"], index=1)

    st.markdown("---")
    if page == "📊 选股策略":
        pool_name = st.selectbox("股票池", list(STOCK_POOLS.keys()))
        all_selectors = list(SELECTORS.keys()) + ML_SELECTORS
        selector_name = st.selectbox("选股策略", all_selectors)
        top_n = st.slider("选股数量", 3, 10, 5)
        run_select = st.button("🔍 运行选股", type="primary", use_container_width=True)
    else:
        st.subheader("标的设置")
        symbol = st.text_input("股票代码", value="600519")
        start_date = st.date_input("起始日期", value=pd.to_datetime("2023-01-01"))
        all_signals = list(SIGNAL_MODELS.keys()) + ML_SIGNAL_MODELS
        signal_name = st.selectbox("信号模型", all_signals)

        st.markdown("---")
        st.subheader("🤖 ML增强")
        use_meta_label = st.checkbox("启用 Meta-Labeling 过滤", value=False,
                                      help="用ML判断规则信号是否可靠，过滤低质量信号")
        use_regime = st.checkbox("启用市场状态过滤", value=False,
                                  help="识别牛市/熊市/震荡，熊市抑制买入信号")

        st.markdown("---")
        st.subheader("回测参数")
        initial_capital = st.number_input("初始资金 (元)", value=100000, step=10000)
        commission = st.slider("佣金费率 (万)", 1.0, 10.0, 2.5, 0.5) / 10000
        slippage = st.slider("滑点 (%)", 0.0, 0.5, 0.1, 0.05) / 100

        st.markdown("---")
        st.markdown("""
        **A股规则**
        - T+1: 当日买入次日方可卖出
        - 印花税: 卖出千分之0.5
        - 涨跌停: 主板±10%, 创业板/科创板±20%
        - 最小交易单位: 100股 (1手)
        """)
        run_bt = st.button("🚀 运行回测", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1: STOCK SELECTION
# ══════════════════════════════════════════════════════════════
if page == "📊 选股策略":
    st.title("📊 A股智能选股")
    st.caption("基于技术指标的多策略选股系统")

    if "run_select" in dir() and run_select:
        pool = STOCK_POOLS[pool_name]

        if selector_name in SELECTORS:
            selector = SELECTORS[selector_name]
            with st.spinner(f"正在运行 {selector_name}，分析 {len(pool)} 只股票..."):
                results = selector(pool, top_n=top_n)
        elif selector_name == "ML多因子选股 (LightGBM)":
            ml_selector = MLFactorSelector(predict_days=5)
            progress_bar = st.progress(0, text="训练ML多因子模型...")
            ml_selector.train(pool, progress_callback=lambda p: progress_bar.progress(p, text=f"训练中... {p:.0%}"))
            progress_bar.empty()
            results = ml_selector.select(pool, top_n=top_n)
            if ml_selector.metrics:
                st.info(f"ML模型训练完成 — {' | '.join(f'{k}: {v}' for k, v in ml_selector.metrics.items())}")
                imp = ml_selector.get_feature_importance()
                if len(imp) > 0:
                    with st.expander("📊 ML因子重要性"):
                        fig = go.Figure(go.Bar(y=imp["feature"].head(15), x=imp["importance"].head(15),
                                                orientation="h", marker_color="teal"))
                        fig.update_layout(height=350, title="Top-15 因子重要性", margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig, use_container_width=True)
        else:
            results = []

        if results:
            st.subheader(f"🏆 {selector_name} — 选股结果")
            res_df = pd.DataFrame(results, columns=["代码", "评分", "理由"])
            res_df.index = range(1, len(res_df) + 1)
            res_df.index.name = "排名"
            st.dataframe(res_df, use_container_width=True)

            # Show charts for top picks
            for i, (sym, score, reason) in enumerate(results[:3]):
                with st.expander(f"#{i+1} {sym} — 评分 {score} | {reason}", expanded=(i == 0)):
                    try:
                        df = fetch_stock(sym)
                        df = add_technical_indicators(df)
                        df_plot = df.tail(120)

                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                            row_heights=[0.5, 0.25, 0.25],
                                            vertical_spacing=0.03)
                        fig.add_trace(go.Candlestick(
                            x=df_plot["date"], open=df_plot["open"],
                            high=df_plot["high"], low=df_plot["low"],
                            close=df_plot["close"], name="K线"), row=1, col=1)
                        for ma, color in [("ma5","#FF9800"),("ma20","#2196F3"),("ma60","#9C27B0")]:
                            if ma in df_plot:
                                fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot[ma],
                                    mode="lines", name=ma.upper(), line=dict(width=1, color=color)),
                                    row=1, col=1)

                        colors = ["#F44336" if r >= 0 else "#4CAF50" for r in df_plot["pct_chg"]]
                        fig.add_trace(go.Bar(x=df_plot["date"], y=df_plot["volume"],
                            marker_color=colors, name="成交量", opacity=0.6), row=2, col=1)

                        fig.add_trace(go.Bar(x=df_plot["date"], y=df_plot["macd_hist"],
                            marker_color=["#F44336" if v > 0 else "#4CAF50" for v in df_plot["macd_hist"]],
                            name="MACD柱"), row=3, col=1)

                        fig.update_layout(height=500, showlegend=False, xaxis_rangeslider_visible=False,
                                           margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"获取 {sym} 数据失败: {e}")
        else:
            st.warning("未找到符合条件的股票")
    else:
        st.info("👈 选择股票池和策略，点击 **运行选股**")

        st.subheader("📋 可用策略")
        for name, func in SELECTORS.items():
            st.markdown(f"- **{name}**: {func.__doc__.strip()}")

        st.subheader("📋 股票池")
        for name, pool in STOCK_POOLS.items():
            st.markdown(f"- **{name}**: {len(pool)} 只 — `{', '.join(pool[:5])}...`")

# ══════════════════════════════════════════════════════════════
# PAGE 2: SIGNAL & BACKTEST
# ══════════════════════════════════════════════════════════════
elif page == "📈 交易信号 & 回测":
    st.title("📈 A股交易信号与回测系统")
    st.caption(f"股票: {symbol} | 信号: {signal_name} | T+1 / 涨跌停 / 交易费用")

    if "run_bt" in dir() and run_bt:
        with st.spinner(f"加载 {symbol} 数据并运行回测..."):
            try:
                df = fetch_stock(symbol, start_date=start_date.strftime("%Y%m%d"))
                df = add_technical_indicators(df)
            except Exception as e:
                st.error(f"获取数据失败: {e}")
                st.stop()

            # Generate signals
            ml_info = {}
            if signal_name == "LSTM 深度学习信号":
                lstm_model = LSTMSignalModel(seq_len=30, epochs=50)
                lstm_model.fit(df)
                signals = lstm_model.predict_signals(df)
                ml_info["LSTM模型"] = lstm_model.metrics
                ml_info["_lstm_proba"] = lstm_model.predict_proba_series(df)
            elif signal_name == "Meta-Labeling 信号过滤":
                from astock.signal_model import composite_signal
                raw_signals = composite_signal(df)
                meta = MetaLabeler(hold_days=5, confidence_threshold=0.55)
                meta.fit(df, raw_signals)
                signals = meta.filter_signals(df, raw_signals)
                ml_info["Meta-Labeling"] = meta.metrics
                ml_info["_meta_imp"] = meta.get_feature_importance()
                ml_info["原始信号数"] = int((raw_signals != 0).sum())
                ml_info["过滤后信号数"] = int((signals != 0).sum())
            else:
                signal_func = SIGNAL_MODELS[signal_name]
                signals = signal_func(df)

            # Apply Meta-Labeling filter (on top of rule-based signals)
            if use_meta_label and signal_name not in ML_SIGNAL_MODELS:
                meta = MetaLabeler(hold_days=5, confidence_threshold=0.55)
                meta.fit(df, signals)
                raw_count = int((signals != 0).sum())
                signals = meta.filter_signals(df, signals)
                ml_info["Meta-Labeling过滤"] = meta.metrics
                ml_info["过滤前信号"] = raw_count
                ml_info["过滤后信号"] = int((signals != 0).sum())

            # Apply regime filter
            regime_info = None
            if use_regime:
                regime_det = MarketRegimeDetector()
                regime_det.fit(df)
                regimes = regime_det.detect(df)
                signals = regime_det.filter_signals_by_regime(signals, regimes)
                regime_info = regime_det.detect_current(df)
                ml_info["市场状态"] = regime_info

            bt_config = BacktestConfig(
                initial_capital=initial_capital,
                commission_rate=commission,
                slippage=slippage,
            )
            bt = run_backtest(df, signals, bt_config)

        # ── Tabs ──
        tab1, tab2, tab3, tab4 = st.tabs(["📊 K线与信号", "💰 回测结果", "📋 交易明细", "📉 风险分析"])

        # TAB 1: Chart with signals
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            latest = df.iloc[-1]
            with col1:
                st.metric("最新价", f"{latest['close']:.2f}",
                           delta=f"{latest['pct_chg']:.2f}%")
            with col2:
                st.metric("信号", "买入 🟢" if signals.iloc[-1] > 0 else
                           ("卖出 🔴" if signals.iloc[-1] < 0 else "观望 ⚪"))
            with col3:
                st.metric("RSI", f"{latest.get('rsi14', 0):.1f}")
            with col4:
                st.metric("MACD", f"{latest.get('macd_hist', 0):.3f}")

            # Latest signal card
            last_sig = signals.iloc[-1]
            if last_sig > 0:
                st.markdown('<div class="signal-buy"><h2>买入信号 🟢</h2></div>', unsafe_allow_html=True)
            elif last_sig < 0:
                st.markdown('<div class="signal-sell"><h2>卖出信号 🔴</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="signal-hold"><h2>观望 ⚪</h2></div>', unsafe_allow_html=True)

            # ML info display
            if ml_info:
                with st.expander("🤖 ML 模型详情", expanded=False):
                    for key, val in ml_info.items():
                        if key.startswith("_"):
                            continue
                        if isinstance(val, dict):
                            st.markdown(f"**{key}**")
                            st.json(val)
                        else:
                            st.markdown(f"**{key}**: {val}")

                    # LSTM probability chart
                    if "_lstm_proba" in ml_info:
                        proba = ml_info["_lstm_proba"]
                        fig_p = go.Figure()
                        fig_p.add_trace(go.Scatter(x=df.tail(120)["date"],
                            y=proba.tail(120), mode="lines", name="P(上涨)",
                            line=dict(color="#E91E63")))
                        fig_p.add_hline(y=0.6, line_dash="dot", line_color="green")
                        fig_p.add_hline(y=0.4, line_dash="dot", line_color="red")
                        fig_p.update_layout(title="LSTM 上涨概率", height=250)
                        st.plotly_chart(fig_p, use_container_width=True)

                    # Meta-labeling feature importance
                    if "_meta_imp" in ml_info:
                        imp = ml_info["_meta_imp"]
                        if len(imp) > 0:
                            fig_i = go.Figure(go.Bar(y=imp["feature"].head(10),
                                x=imp["importance"].head(10), orientation="h", marker_color="teal"))
                            fig_i.update_layout(title="Meta-Labeling 特征重要性", height=300)
                            st.plotly_chart(fig_i, use_container_width=True)

            # Regime display
            if regime_info:
                rc = regime_info["颜色"]
                st.markdown(f'<div style="background:{rc};color:white;padding:10px;border-radius:8px;text-align:center;">'
                            f'<b>市场状态: {regime_info["当前状态"]}</b> — {regime_info["建议"]}</div>',
                            unsafe_allow_html=True)

            # K-line chart with signals
            df_plot = df.tail(min(250, len(df))).copy()
            sig_plot = signals.reindex(df_plot.index)

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                row_heights=[0.4, 0.2, 0.2, 0.2],
                                vertical_spacing=0.02,
                                subplot_titles=["K线 + 信号", "成交量", "MACD", "RSI"])

            fig.add_trace(go.Candlestick(
                x=df_plot["date"], open=df_plot["open"], high=df_plot["high"],
                low=df_plot["low"], close=df_plot["close"], name="K线"), row=1, col=1)

            for ma, color in [("ma5","orange"),("ma20","blue"),("ma60","purple")]:
                if ma in df_plot:
                    fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot[ma],
                        mode="lines", name=ma.upper(), line=dict(width=1, color=color)),
                        row=1, col=1)

            buy_dates = df_plot[sig_plot > 0]["date"]
            buy_prices = df_plot[sig_plot > 0]["low"] * 0.98
            sell_dates = df_plot[sig_plot < 0]["date"]
            sell_prices = df_plot[sig_plot < 0]["high"] * 1.02
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#4CAF50"),
                name="买入"), row=1, col=1)
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#F44336"),
                name="卖出"), row=1, col=1)

            vol_colors = ["#F44336" if r >= 0 else "#4CAF50" for r in df_plot["pct_chg"]]
            fig.add_trace(go.Bar(x=df_plot["date"], y=df_plot["volume"],
                marker_color=vol_colors, opacity=0.6, name="成交量"), row=2, col=1)

            fig.add_trace(go.Bar(x=df_plot["date"], y=df_plot["macd_hist"],
                marker_color=["#F44336" if v > 0 else "#4CAF50" for v in df_plot["macd_hist"]],
                name="MACD"), row=3, col=1)

            if "rsi14" in df_plot:
                fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["rsi14"],
                    mode="lines", name="RSI", line=dict(color="#FF9800")), row=4, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=4, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=4, col=1)

            fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: Backtest results
        with tab2:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                color = "normal" if bt["总收益率(%)"] >= 0 else "inverse"
                st.metric("总收益率", f"{bt['总收益率(%)']:.2f}%")
            with col2:
                st.metric("胜率", f"{bt['胜率(%)']:.1f}%")
            with col3:
                st.metric("Sharpe", f"{bt['年化Sharpe']:.3f}")
            with col4:
                st.metric("最大回撤", f"{bt['最大回撤(%)']:.2f}%")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最终资金", f"¥{bt['最终资金']:,.0f}")
            with col2:
                st.metric("交易次数", bt["交易次数"])
            with col3:
                st.metric("买入持有", f"{bt['买入持有(%)']:.2f}%")
            with col4:
                st.metric("总交易费用", f"¥{bt['总交易费用']:,.0f}")

            # Equity curve
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3],
                                subplot_titles=["净值曲线", "持仓市值"])
            fig.add_trace(go.Scatter(x=bt["_dates"], y=bt["_equity"],
                mode="lines", name="策略净值", line=dict(color="#E91E63", width=2),
                fill="tozeroy", fillcolor="rgba(233,30,99,0.1)"), row=1, col=1)
            fig.add_hline(y=initial_capital, line_dash="dot", line_color="gray", row=1, col=1)

            # Buy-and-hold
            bh_equity = initial_capital * df["close"].values / df["close"].values[0]
            fig.add_trace(go.Scatter(x=bt["_dates"], y=bh_equity,
                mode="lines", name="买入持有", line=dict(color="gray", width=1.5, dash="dash")),
                row=1, col=1)

            fig.add_trace(go.Scatter(x=bt["_dates"], y=bt["_position"],
                mode="lines", name="持仓", fill="tozeroy",
                line=dict(color="#2196F3")), row=2, col=1)

            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary = {k: v for k, v in bt.items() if not k.startswith("_")}
            st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

        # TAB 3: Trade details
        with tab3:
            st.subheader("📋 交易记录")
            if bt["_trades"]:
                trade_rows = []
                for t in bt["_trades"]:
                    trade_rows.append({
                        "日期": t.date, "方向": t.action,
                        "价格": t.price, "数量": t.shares,
                        "金额": t.amount, "佣金": t.commission,
                        "印花税": t.stamp_duty, "过户费": t.transfer_fee,
                        "总费用": t.total_cost,
                    })
                st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)

                # Fee breakdown
                st.subheader("💸 费用分析")
                total_comm = sum(t.commission for t in bt["_trades"])
                total_stamp = sum(t.stamp_duty for t in bt["_trades"])
                total_transfer = sum(t.transfer_fee for t in bt["_trades"])
                fee_df = pd.DataFrame([
                    {"费用类型": "佣金", "金额": round(total_comm, 2)},
                    {"费用类型": "印花税 (卖出)", "金额": round(total_stamp, 2)},
                    {"费用类型": "过户费", "金额": round(total_transfer, 2)},
                    {"费用类型": "合计", "金额": round(total_comm + total_stamp + total_transfer, 2)},
                ])
                st.dataframe(fee_df, use_container_width=True, hide_index=True)
            else:
                st.info("本期无交易记录")

        # TAB 4: Risk analysis
        with tab4:
            st.subheader("📉 风险分析")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["回撤曲线", "日收益分布"])
            fig.add_trace(go.Scatter(x=bt["_dates"], y=-bt["_drawdown"],
                mode="lines", fill="tozeroy", name="回撤 %",
                line=dict(color="#F44336")), row=1, col=1)

            daily_ret = np.diff(bt["_equity"]) / (bt["_equity"][:-1] + 1e-10) * 100
            fig.add_trace(go.Histogram(x=daily_ret, nbinsx=50, name="日收益 %",
                marker_color="#2196F3", opacity=0.7), row=2, col=1)
            fig.add_vline(x=0, line_dash="dot", line_color="black", row=2, col=1)

            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Risk metrics
            if len(daily_ret) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("日均收益", f"{daily_ret.mean():.4f}%")
                with col2:
                    st.metric("日波动率", f"{daily_ret.std():.4f}%")
                with col3:
                    var_95 = np.percentile(daily_ret, 5)
                    st.metric("VaR (95%)", f"{var_95:.4f}%")

        st.success(f"✅ 回测完成 | {symbol} | {signal_name} | "
                   f"收益: {bt['总收益率(%)']:.2f}% | 胜率: {bt['胜率(%)']:.1f}%")

    else:
        st.info("👈 输入股票代码，选择信号模型，点击 **运行回测**")

        st.subheader("📋 可用信号模型")
        for name in SIGNAL_MODELS:
            st.markdown(f"- **{name}**")

        st.subheader("🇨🇳 A股交易规则")
        st.markdown("""
        | 规则 | 说明 |
        |---|---|
        | T+1 | 当日买入的股票次日方可卖出 |
        | 涨跌停 | 主板 ±10%，创业板/科创板 ±20% |
        | 最小单位 | 100股 (1手) |
        | 佣金 | 默认万2.5，最低5元 |
        | 印花税 | 卖出千分之0.5 |
        | 过户费 | 双向万分之0.1 |
        """)
