
import pandas as pd
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from src.data_loader import load_market_data
from src.market_structure import detect_market_structure
from src.bos_detector import detect_bos
from src.liquidity_sweep import detect_liquidity_sweeps
from src.backtester import run_backtest
from src.advanced_backtester import run_advanced_backtest
from src.risk_simulator import simulate_risk_adjusted_equity
from src.risk_manager import calculate_risk
from src.performance import analyze_performance
from src.data_quality import check_data_quality
from src.ml_features import create_ml_features
from src.ml_model import train_ml_model, compare_ml_models


st.set_page_config(
    page_title="AURUM AI",
    page_icon="📈",
    layout="wide"
)

st.title("AURUM AI")
st.subheader("XAUUSD Market Analysis")

project_dir = Path(__file__).resolve().parent
csv_path = project_dir / "data" / "xauusd_1m.csv"


try:
    # ----------------------------------
    # LOAD HISTORICAL DATA
    # ----------------------------------

    df = load_market_data(csv_path)

    if df.empty:
        st.error("The historical dataset is empty.")
        st.stop()

    # ----------------------------------
    # DATA QUALITY
    # ----------------------------------

    st.header("Market Data Quality")

    quality = check_data_quality(df)

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    quality_col1.metric(
        "Total Candles",
        quality["total_candles"]
    )

    quality_col2.metric(
        "Duplicate Timestamps",
        quality["duplicate_timestamps"]
    )

    quality_col3.metric(
        "Missing Minutes",
        quality["missing_minutes"]
    )

    st.write("### Data Validation Details")

    st.dataframe(
        pd.DataFrame([quality]),
        use_container_width=True
    )

    # ----------------------------------
    # MARKET ANALYSIS
    # ----------------------------------

    df = detect_market_structure(df)
    df = detect_bos(df)
    df = detect_liquidity_sweeps(df)

    # ----------------------------------
    # MAIN MARKET METRICS
    # ----------------------------------

    st.success("Market data loaded successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Candles", len(df))

    col2.metric(
        "Latest Close",
        f"${df['close'].iloc[-1]:,.2f}"
    )

    col3.metric(
        "Highest Price",
        f"${df['high'].max():,.2f}"
    )

    # ----------------------------------
    # CANDLESTICK CHART
    # ----------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="XAUUSD"
        )
    )

    swing_highs = df[df["swing_high"]]

    fig.add_trace(
        go.Scatter(
            x=swing_highs["timestamp"],
            y=swing_highs["high"],
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                size=13,
                color="red"
            ),
            name="Swing High"
        )
    )

    swing_lows = df[df["swing_low"]]

    fig.add_trace(
        go.Scatter(
            x=swing_lows["timestamp"],
            y=swing_lows["low"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=13,
                color="lime"
            ),
            name="Swing Low"
        )
    )

    bullish = df[df["bullish_bos"]]

    fig.add_trace(
        go.Scatter(
            x=bullish["timestamp"],
            y=bullish["close"],
            mode="markers+text",
            marker=dict(
                color="cyan",
                size=13,
                symbol="diamond"
            ),
            text=["BOS ↑"] * len(bullish),
            textposition="top center",
            name="Bullish BOS"
        )
    )

    bearish = df[df["bearish_bos"]]

    fig.add_trace(
        go.Scatter(
            x=bearish["timestamp"],
            y=bearish["close"],
            mode="markers+text",
            marker=dict(
                color="orange",
                size=13,
                symbol="diamond"
            ),
            text=["BOS ↓"] * len(bearish),
            textposition="bottom center",
            name="Bearish BOS"
        )
    )

    bullish_sweeps = df[df["bullish_sweep"]]

    fig.add_trace(
        go.Scatter(
            x=bullish_sweeps["timestamp"],
            y=bullish_sweeps["low"],
            mode="markers",
            marker=dict(
                color="lime",
                size=16,
                symbol="star"
            ),
            name="Bullish Liquidity Sweep"
        )
    )

    bearish_sweeps = df[df["bearish_sweep"]]

    fig.add_trace(
        go.Scatter(
            x=bearish_sweeps["timestamp"],
            y=bearish_sweeps["high"],
            mode="markers",
            marker=dict(
                color="magenta",
                size=16,
                symbol="star"
            ),
            name="Bearish Liquidity Sweep"
        )
    )

    fig.update_layout(
        title="XAUUSD Candlestick Chart",
        template="plotly_dark",
        height=550,
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------
    # MARKET STRUCTURE ANALYSIS
    # ----------------------------------

    st.subheader("Historical Market Data")
    st.subheader("Break of Structure Analysis")

    bos_col1, bos_col2 = st.columns(2)

    bos_col1.metric(
        "Bullish BOS",
        int(df["bullish_bos"].sum())
    )

    bos_col2.metric(
        "Bearish BOS",
        int(df["bearish_bos"].sum())
    )

    st.subheader("Liquidity Sweep Analysis")

    sweep_col1, sweep_col2 = st.columns(2)

    sweep_col1.metric(
        "Bullish Sweeps",
        int(df["bullish_sweep"].sum())
    )

    sweep_col2.metric(
        "Bearish Sweeps",
        int(df["bearish_sweep"].sum())
    )

    st.dataframe(df, use_container_width=True)

    # ----------------------------------
    # RISK MANAGEMENT SIMULATOR
    # ----------------------------------

    st.divider()
    st.header("Risk Management Simulator")

    risk_balance = st.number_input(
        "Hypothetical Account Balance (USD)",
        min_value=1.0,
        value=5000.0
    )

    risk_percentage = st.number_input(
        "Hypothetical Risk (%)",
        min_value=0.1,
        max_value=100.0,
        value=1.0
    )

    risk_entry = st.number_input(
        "Example Entry Price",
        min_value=0.01,
        value=4300.0
    )

    risk_stop = st.number_input(
        "Example Stop-Loss Price",
        min_value=0.01,
        value=4295.0
    )

    try:
        risk_result = calculate_risk(
            balance=risk_balance,
            risk_percent=risk_percentage,
            entry_price=risk_entry,
            stop_loss=risk_stop
        )

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        risk_col1.metric(
            "Hypothetical Risk Budget",
            f"${risk_result.risk_amount:.2f}"
        )

        risk_col2.metric(
            "Position Units",
            f"{risk_result.position_units:.4f}"
        )

        risk_col3.metric(
            "Estimated Loss",
            f"${risk_result.estimated_loss:.2f}"
        )

        st.caption(
            "Simulation only. Assumes linear USD price "
            "exposure and excludes execution costs."
        )

    except ValueError as error:
        st.warning(str(error))

    # ----------------------------------
    # ADVANCED STRATEGY BACKTESTING
    # ----------------------------------

    st.divider()
    st.header("Advanced Strategy Backtesting")

    st.write(
        "Research simulation with stop-loss, take-profit "
        "and multi-candle holding."
    )

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        adv_stop = st.number_input(
            "Stop-Loss Distance (price units)",
            min_value=0.1,
            value=5.0,
            step=0.5
        )

        adv_rr = st.number_input(
            "Reward-to-Risk Ratio",
            min_value=0.1,
            value=2.0,
            step=0.5
        )

        adv_holding = st.number_input(
            "Maximum Holding Candles",
            min_value=1,
            max_value=500,
            value=30
        )

    with adv_col2:
        adv_spread = st.number_input(
            "Advanced Spread",
            min_value=0.0,
            value=0.30,
            step=0.05
        )

        adv_slippage = st.number_input(
            "Advanced Slippage",
            min_value=0.0,
            value=0.10,
            step=0.05
        )

    advanced_trades = run_advanced_backtest(
        df,
        stop_distance=adv_stop,
        reward_ratio=adv_rr,
        spread=adv_spread,
        slippage=adv_slippage,
        max_holding_candles=adv_holding
    )

    if advanced_trades.empty:
        st.warning(
            "No advanced trades were generated "
            "for these settings."
        )

    else:
        total_advanced = len(advanced_trades)

        winning_advanced = (
            advanced_trades["pnl_points"] > 0
        ).sum()

        advanced_win_rate = (
            winning_advanced / total_advanced * 100
        )

        advanced_net = advanced_trades[
            "pnl_points"
        ].sum()

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Advanced Trades",
            total_advanced
        )

        metric2.metric(
            "Advanced Win Rate",
            f"{advanced_win_rate:.2f}%"
        )

        metric3.metric(
            "Net Price Points",
            f"{advanced_net:.2f}"
        )

        st.subheader("Advanced Exit Reasons")

        exit_counts = advanced_trades[
            "exit_reason"
        ].value_counts()

        st.bar_chart(exit_counts)

        # ----------------------------------
        # ADVANCED PERFORMANCE
        # ----------------------------------

        st.subheader("Advanced Equity Analysis")

        advanced_performance, advanced_metrics = (
            analyze_performance(
                advanced_trades,
                starting_equity=5000.0,
                units=1.0
            )
        )

        eq_col1, eq_col2, eq_col3 = st.columns(3)

        eq_col1.metric(
            "Final Simulated Equity",
            f"${5000 + advanced_metrics['net_profit']:.2f}"
        )

        eq_col2.metric(
            "Maximum Drawdown",
            f"${advanced_metrics['max_drawdown']:.2f}"
        )

        eq_col3.metric(
            "Maximum Drawdown %",
            f"{advanced_metrics['max_drawdown_percent']:.2f}%"
        )

        st.subheader("Advanced Equity Curve")

        equity_data = advanced_performance.set_index(
            "exit_time"
        )["equity"]

        st.line_chart(equity_data)

        st.subheader("Advanced Drawdown Curve")

        drawdown_data = advanced_performance.set_index(
            "exit_time"
        )["drawdown_percent"]

        st.area_chart(drawdown_data)

        st.caption(
            "Hypothetical results using one fixed unit per "
            "trade and a $5,000 starting balance. "
            "Not actual trading performance."
        )

        # ----------------------------------
        # RISK-ADJUSTED EQUITY SIMULATION
        # ----------------------------------

        st.subheader("Risk-Adjusted Equity Simulation")

        simulated_balance = st.number_input(
            "Simulation Starting Balance (USD)",
            min_value=100.0,
            value=5000.0,
            step=100.0
        )

        simulated_risk = st.number_input(
            "Simulation Risk Budget (%)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        )

        try:
            risk_trades, risk_metrics = (
                simulate_risk_adjusted_equity(
                    advanced_trades,
                    starting_balance=simulated_balance,
                    risk_percent=simulated_risk
                )
            )

            risk_metric1, risk_metric2, risk_metric3 = (
                st.columns(3)
            )

            risk_metric1.metric(
                "Final Simulated Balance",
                f"${risk_metrics['final_balance']:.2f}"
            )

            risk_metric2.metric(
                "Simulated Net P&L",
                f"${risk_metrics['net_profit']:.2f}"
            )

            risk_metric3.metric(
                "Maximum Drawdown %",
                f"{risk_metrics['max_drawdown_percent']:.2f}%"
            )

            if not risk_trades.empty:
                st.subheader("Risk-Adjusted Equity Curve")

                st.line_chart(
                    risk_trades.set_index("exit_time")["equity"]
                )

                st.subheader("Risk-Adjusted Drawdown")

                st.area_chart(
                    risk_trades.set_index("exit_time")[
                        "drawdown_percent"
                    ]
                )

                st.subheader("Risk-Adjusted Trade Results")

                st.dataframe(
                    risk_trades,
                    use_container_width=True
                )

            st.caption(
                "Educational simulation using hypothetical "
                "risk budgets. Position sizes are recalculated "
                "after each completed trade. Results exclude "
                "margin constraints and additional fees."
            )

        except (ValueError, KeyError) as error:
            st.warning(
                f"Risk simulation unavailable: {error}"
            )

        st.subheader("Advanced Trade History")

        st.dataframe(
            advanced_trades,
            use_container_width=True
        )

        st.download_button(
            "Download Advanced Trades CSV",
            data=advanced_trades.to_csv(index=False),
            file_name="aurum_advanced_backtest.csv",
            mime="text/csv"
        )

    st.caption(
        "Historical simulation only. Minute candles "
        "cannot establish exact intrabar execution order. "
        "Results are not evidence of future profitability."
    )

    # ----------------------------------
    # BASIC STRATEGY BACKTESTING
    # ----------------------------------

    st.divider()
    st.header("Strategy Backtesting")

    st.write("### Backtesting Settings")

    spread = st.number_input(
        "Assumed Spread (price units)",
        min_value=0.0,
        value=0.30,
        step=0.05
    )

    slippage = st.number_input(
        "Assumed Slippage per execution (price units)",
        min_value=0.0,
        value=0.10,
        step=0.05
    )

    trades = run_backtest(
        df,
        spread=spread,
        slippage=slippage
    )

    if trades.empty:
        st.info(
            "No simulated trades found. "
            "Try a larger historical dataset."
        )

    else:
        total_trades = len(trades)

        wins = (trades["pnl_points"] > 0).sum()
        win_rate = wins / total_trades * 100

        total_pnl = trades["pnl_points"].sum()

        bt_col1, bt_col2, bt_col3 = st.columns(3)

        bt_col1.metric("Simulated Trades", total_trades)

        bt_col2.metric(
            "Win Rate",
            f"{win_rate:.1f}%"
        )

        bt_col3.metric(
            "Net Price Points",
            f"{total_pnl:.2f}"
        )

        st.subheader("Performance Analysis")

        performance_df, metrics = analyze_performance(
            trades,
            starting_equity=5000.0,
            units=1.0
        )

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        perf_col1.metric(
            "Net Simulated Profit",
            f"${metrics['net_profit']:.2f}"
        )

        perf_col2.metric(
            "Maximum Drawdown",
            f"${metrics['max_drawdown']:.2f}"
        )

        perf_col3.metric(
            "Maximum Drawdown %",
            f"{metrics['max_drawdown_percent']:.2f}%"
        )

        st.subheader("Simulated Equity Curve")

        equity_chart = performance_df.set_index(
            "entry_time"
        )["equity"]

        st.line_chart(equity_chart)

        st.caption(
            "Hypothetical results using fixed position units. "
            "Not actual account performance."
        )

        st.subheader("Simulated Trade History")

        st.dataframe(trades)

        st.subheader("Cumulative Price-Point Results")

        trades = trades.copy()

        trades["cumulative_pnl"] = (
            trades["pnl_points"].cumsum()
        )

        st.line_chart(
            trades.set_index("entry_time")["cumulative_pnl"]
        )

        st.caption(
            "Educational backtest. Interpret results using "
            "the spread and slippage assumptions above."
        )

    # ----------------------------------
    # MACHINE LEARNING LAB
    # ----------------------------------

    st.divider()

    st.header("AURUM AI — Machine Learning Lab")

    st.write(
        "Train and evaluate machine learning models "
        "using historical XAUUSD candles."
    )

    st.subheader("Logistic Regression Training")

    if st.button("Train ML Model", type="primary"):

        with st.spinner("Training ML model..."):

            ml_data = create_ml_features(df)

            ml_model, ml_metrics, ml_predictions = (
                train_ml_model(ml_data)
            )

            st.success("ML model trained and evaluated!")

            ml_col1, ml_col2, ml_col3 = st.columns(3)

            ml_col1.metric(
                "Test Accuracy",
                f"{ml_metrics['accuracy'] * 100:.2f}%"
            )

            ml_col2.metric(
                "Balanced Accuracy",
                f"{ml_metrics['balanced_accuracy'] * 100:.2f}%"
            )

            ml_col3.metric(
                "Baseline Accuracy",
                f"{ml_metrics['baseline_accuracy'] * 100:.2f}%"
            )

            st.write("### Training Information")

            st.write(
                f"Training rows: {ml_metrics['training_rows']}"
            )

            st.write(
                f"Testing rows: {ml_metrics['testing_rows']}"
            )

            st.write("### Model Predictions")

            st.dataframe(
                ml_predictions.tail(20),
                use_container_width=True
            )

            st.caption(
                "Experimental classification research only. "
                "Predictions are not trading recommendations. "
                "Accuracy does not establish profitability."
            )

    # ----------------------------------
    # ML MODEL COMPARISON
    # ----------------------------------

    st.divider()

    st.subheader("ML Model Comparison")

    st.write(
        "Compare Logistic Regression against "
        "Random Forest using the same historical data."
    )

    if st.button("Compare ML Models"):

        with st.spinner("Comparing ML algorithms..."):

            comparison_data = create_ml_features(df)

            comparison_results = compare_ml_models(
                comparison_data
            )

            st.success("Model comparison completed!")

            st.dataframe(
                comparison_results.style.format({
                    "Accuracy": "{:.2%}",
                    "Balanced Accuracy": "{:.2%}",
                    "Baseline Accuracy": "{:.2%}"
                }),
                use_container_width=True
            )

            st.write("### Balanced Accuracy Comparison")

            st.bar_chart(
                comparison_results.set_index("Model")[
                    "Balanced Accuracy"
                ]
            )

            st.caption(
                "Both models use the same chronological "
                "training and testing periods. Results "
                "are experimental and do not establish "
                "trading profitability."
            )


except Exception as error:
    st.error(f"Error loading dashboard: {error}")
