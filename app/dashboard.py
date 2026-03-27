"""
Asset Price Prediction Dashboard — Streamlit Application.

A professional trading-oriented prediction tool for Chinese/US government bonds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from data_fetcher import get_series, list_assets, fetch_bond_data
from predictor import Predictor
from backtester import backtest

st.set_page_config(page_title="债券收益率预测系统", page_icon="📈", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white; text-align: center;
        margin: 5px 0;
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.8; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 28px; }
    .signal-buy { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .signal-sell { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .signal-hold { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ 参数设置")

    asset = st.selectbox("选择资产", list_assets(), index=2)
    forecast_days = st.slider("预测天数", 1, 20, 5)
    signal_threshold = st.slider("信号阈值 (bp)", 0.0, 5.0, 0.5, 0.1) / 1000
    train_ratio = st.slider("训练集比例", 0.7, 0.95, 0.85, 0.05)

    st.markdown("---")
    st.markdown("### 回测参数")
    notional = st.number_input("名义本金 (万元)", value=100, step=10) * 10000
    bt_threshold = st.slider("回测开仓阈值 (bp)", 0.0, 3.0, 0.0, 0.1) / 1000

    st.markdown("---")
    run_btn = st.button("🚀 运行预测", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
st.title("📈 债券收益率预测与交易信号系统")
st.caption(f"数据源: 东方财富 | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if run_btn:
    with st.spinner("加载数据 & 训练模型中..."):
        series = get_series(asset)
        pred = Predictor(series, train_ratio=train_ratio)
        pred.train()
        eval_df = pred.evaluate()
        forecast_df, forecast_preds = pred.predict_next(forecast_days)
        signal = pred.generate_signals(forecast_preds, threshold=signal_threshold)

    # ── Tab Layout ──
    tab1, tab2, tab3, tab4 = st.tabs(["📊 市场概览", "🔮 预测结果", "📋 模型评估", "💰 回测分析"])

    # ══════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最新收益率", f"{series.iloc[-1]:.4f}%",
                       delta=f"{series.iloc[-1]-series.iloc[-2]:.4f}")
        with col2:
            st.metric("5日变化", f"{series.iloc[-1]-series.iloc[-6]:.4f}",
                       delta=f"{(series.iloc[-1]-series.iloc[-6]):.4f}")
        with col3:
            st.metric("20日波动率", f"{series.diff().tail(20).std():.4f}")
        with col4:
            st.metric("数据点数", f"{len(series):,}")

        # Price chart with plotly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=[f"{asset} 收益率", "每日变化 Δy"])
        fig.add_trace(go.Scatter(x=series.index, y=series.values,
                                  mode="lines", name="收益率",
                                  line=dict(color="#2196F3", width=1.5)), row=1, col=1)
        # Mark test period
        test_start = series.index[int(len(series) * train_ratio)]
        fig.add_vrect(x0=test_start, x1=series.index[-1],
                       fillcolor="rgba(255,152,0,0.1)", line_width=0, row=1, col=1)

        delta = series.diff().dropna()
        colors = ["#4CAF50" if d > 0 else "#F44336" for d in delta.values]
        fig.add_trace(go.Bar(x=delta.index, y=delta.values, name="Δy",
                              marker_color=colors, opacity=0.6), row=2, col=1)

        fig.update_layout(height=500, showlegend=False,
                           margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 2: PREDICTIONS
    # ══════════════════════════════════════════════════════════
    with tab2:
        # Trading signal card
        if signal:
            sig_class = "signal-buy" if "多" in signal["方向"] else (
                "signal-sell" if "空" in signal["方向"] else "signal-hold")
            st.markdown(f"""
            <div class="metric-card {sig_class}" style="margin-bottom: 20px;">
                <h3>交易信号 ({forecast_days}日)</h3>
                <h1>{signal['方向']}</h1>
                <p>预期变化: {signal['预期变化']:.4f} | 信心: {signal['信心水平']}
                | 一致性: {signal['模型一致性']}
                | 看多: {signal['看多模型']} / 看空: {signal['看空模型']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Forecast chart
        fig = go.Figure()
        last_n = min(60, len(series))
        hist = series.iloc[-last_n:]
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                                  name="历史", line=dict(color="#2196F3", width=2)))

        future_dates = pd.bdate_range(series.index[-1] + pd.tseries.offsets.BDay(1),
                                       periods=forecast_days)
        model_colors = {"Ridge": "#4CAF50", "XGBoost": "#FF9800", "LightGBM": "#E91E63",
                         "RandomForest": "#9C27B0", "ARIMA": "#00BCD4", "集成(均值)": "#F44336"}

        for name, v in forecast_preds.items():
            c = model_colors.get(name, "#607D8B")
            dash = "solid" if name == "集成(均值)" else "dash"
            width = 3 if name == "集成(均值)" else 1.5
            fig.add_trace(go.Scatter(
                x=future_dates, y=v["level"], mode="lines+markers",
                name=name, line=dict(color=c, width=width, dash=dash),
                marker=dict(size=4)))

        fig.add_vline(x=series.index[-1], line_dash="dot", line_color="gray")
        fig.update_layout(title=f"{asset} — 未来 {forecast_days} 日预测",
                           height=450, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("📋 预测明细")
        display_cols = ["日期"] + [c for c in forecast_df.columns if "预测" in c]
        st.dataframe(forecast_df[display_cols], use_container_width=True, hide_index=True)

        # Delta predictions
        st.subheader("📉 每日变化预测")
        delta_cols = ["日期"] + [c for c in forecast_df.columns if "Δ" in c]
        st.dataframe(forecast_df[delta_cols], use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3: MODEL EVALUATION
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🏆 模型排行榜 (测试集)")
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        # Performance charts
        if pred.results:
            best_name = eval_df.iloc[0]["模型"]
            best_res = pred.results[best_name]
            test_dates = pred.test_level.index[:len(best_res["lvl_true"])]

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_dates, y=best_res["lvl_true"],
                                          mode="lines", name="实际", line=dict(color="black", width=2)))
                fig.add_trace(go.Scatter(x=test_dates, y=best_res["lvl_pred"],
                                          mode="lines", name=f"{best_name} 预测",
                                          line=dict(color="#E91E63", width=2, dash="dash")))
                fig.update_layout(title=f"最优模型: {best_name}", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Direction accuracy comparison
                models_dir = []
                for name, res in pred.results.items():
                    models_dir.append({"模型": name, "方向准确率": res["dir_acc"]})
                dir_df = pd.DataFrame(models_dir).sort_values("方向准确率", ascending=True)
                fig = go.Figure(go.Bar(
                    y=dir_df["模型"], x=dir_df["方向准确率"], orientation="h",
                    marker_color=["#4CAF50" if v > 50 else "#F44336" for v in dir_df["方向准确率"]]))
                fig.add_vline(x=50, line_dash="dot", line_color="gray")
                fig.update_layout(title="方向准确率对比", height=350)
                st.plotly_chart(fig, use_container_width=True)

            # All models overlay
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=best_res["lvl_true"],
                                      mode="lines", name="实际", line=dict(color="black", width=2.5)))
            for name, res in pred.results.items():
                fig.add_trace(go.Scatter(
                    x=test_dates, y=res["lvl_pred"], mode="lines", name=name,
                    line=dict(width=1.2), opacity=0.7))
            fig.update_layout(title="所有模型预测 vs 实际", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4: BACKTESTING
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.subheader("💰 回测分析")

        if pred.results:
            bt_results = {}
            for name, res in pred.results.items():
                test_dates_bt = pred.test_level.index[:len(res["delta_true"])]
                bt = backtest(res["delta_true"], res["delta_pred"], test_dates_bt,
                              threshold=bt_threshold, notional=notional)
                bt_results[name] = bt

            # Summary table
            bt_summary = []
            for name, bt in bt_results.items():
                bt_summary.append({
                    "模型": name,
                    "总收益": bt["总收益"],
                    "胜率(%)": bt["胜率(%)"],
                    "年化Sharpe": bt["年化Sharpe"],
                    "最大回撤": bt["最大回撤"],
                    "超额收益": bt["超额收益"],
                })
            bt_df = pd.DataFrame(bt_summary).sort_values("总收益", ascending=False)
            st.dataframe(bt_df, use_container_width=True, hide_index=True)

            # Best model backtest chart
            best_bt_name = bt_df.iloc[0]["模型"]
            best_bt = bt_results[best_bt_name]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总收益 (元)", f"{best_bt['总收益']:,.0f}")
            with col2:
                st.metric("胜率", f"{best_bt['胜率(%)']:.1f}%")
            with col3:
                st.metric("Sharpe", f"{best_bt['年化Sharpe']:.3f}")
            with col4:
                st.metric("最大回撤", f"{best_bt['最大回撤']:,.0f}")

            # PnL chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4],
                                subplot_titles=["累积收益", "每日持仓"])
            fig.add_trace(go.Scatter(
                x=best_bt["_dates"], y=best_bt["_cum_pnl"],
                mode="lines", name=f"{best_bt_name} 策略",
                line=dict(color="#E91E63", width=2), fill="tozeroy"), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=best_bt["_dates"], y=best_bt["_bh_cum"],
                mode="lines", name="买入持有",
                line=dict(color="#9E9E9E", width=1.5, dash="dash")), row=1, col=1)

            pos_colors = ["#4CAF50" if p > 0 else ("#F44336" if p < 0 else "#E0E0E0")
                          for p in best_bt["_positions"]]
            fig.add_trace(go.Bar(
                x=best_bt["_dates"], y=best_bt["_positions"],
                name="持仓", marker_color=pos_colors), row=2, col=1)

            fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # All models PnL comparison
            fig = go.Figure()
            for name, bt in sorted(bt_results.items(), key=lambda x: -x[1]["总收益"]):
                fig.add_trace(go.Scatter(
                    x=bt["_dates"], y=bt["_cum_pnl"], mode="lines",
                    name=f"{name} ({bt['总收益']:,.0f})", line=dict(width=1.5)))
            fig.add_trace(go.Scatter(
                x=best_bt["_dates"], y=best_bt["_bh_cum"],
                mode="lines", name="买入持有",
                line=dict(color="black", width=2, dash="dash")))
            fig.update_layout(title="所有模型回测收益对比", height=400)
            st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ 分析完成 | 资产: {asset} | 模型数: {len(pred.models)} | "
               f"预测: {forecast_days}天 | 信号: {signal.get('方向', 'N/A')}")

else:
    st.info("👈 请在左侧选择资产和参数，然后点击 **运行预测**")

    # Show available assets
    st.subheader("📋 可选资产")
    df = fetch_bond_data()
    col_map = {c: c for c in df.columns if "收益率" in c}
    latest = df.iloc[-1]
    asset_info = []
    for asset_name in list_assets():
        from data_fetcher import ASSET_MAP
        col = ASSET_MAP[asset_name]
        val = latest.get(col, None)
        if pd.notna(val):
            asset_info.append({"资产": asset_name, "最新收益率": f"{val:.4f}%",
                                "日期": latest["日期"]})
    st.dataframe(pd.DataFrame(asset_info), use_container_width=True, hide_index=True)

    st.markdown("""
    ### 📖 使用说明
    1. **选择资产**: 从左侧下拉框选择要预测的国债品种
    2. **设置参数**: 调整预测天数、信号阈值等
    3. **运行预测**: 点击按钮，系统将自动训练 5 个模型并生成预测
    4. **查看结果**: 浏览 4 个标签页获取市场概览、预测、评估和回测信息

    ### 🤖 算法模型
    - **Ridge 回归** — 正则化线性模型，适合捕捉线性趋势
    - **XGBoost** — 梯度提升树，擅长非线性模式
    - **LightGBM** — 轻量级梯度提升，高效准确
    - **Random Forest** — 随机森林，稳健的方向预测
    - **ARIMA** — 经典时序模型，自动阶数选择

    ### ⚠️ 风险提示
    > 本系统仅供研究参考，不构成任何投资建议。金融市场存在不确定性，
    > 模型预测可能失效。请在充分了解风险的前提下做出投资决策。
    """)
