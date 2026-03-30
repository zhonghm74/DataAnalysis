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
from astock.stock_selector import SELECTORS
from astock.signal_model import SIGNAL_MODELS
from astock.backtester import run_backtest, BacktestConfig

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
        selector_name = st.selectbox("选股策略", list(SELECTORS.keys()))
        top_n = st.slider("选股数量", 3, 10, 5)
        run_select = st.button("🔍 运行选股", type="primary", use_container_width=True)
    else:
        st.subheader("标的设置")
        symbol = st.text_input("股票代码", value="600519")
        start_date = st.date_input("起始日期", value=pd.to_datetime("2023-01-01"))
        signal_name = st.selectbox("信号模型", list(SIGNAL_MODELS.keys()))

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
        selector = SELECTORS[selector_name]

        with st.spinner(f"正在运行 {selector_name}，分析 {len(pool)} 只股票..."):
            results = selector(pool, top_n=top_n)

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

            signal_func = SIGNAL_MODELS[signal_name]
            signals = signal_func(df)

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
