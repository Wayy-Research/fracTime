"""
Exogenous Predictors Test and Model Comparison

This script demonstrates:
1. The new interactive path density visualization
2. Exogenous predictors integration with fractal forecasting
3. Performance comparison against baseline models

Uses wrdata or yfinance to pull time series data for testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# FracTime imports
import fractime as ft
from fractime.baselines.arima import ARIMAForecaster
from fractime.baselines.ets import ETSForecaster
from fractime.forecasting.ml import RandomForestForecaster, XGBoostForecaster


def fetch_test_data(tickers=['SPY', 'VIX', 'TLT', 'GLD'], start_date='2020-01-01'):
    """
    Fetch test data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for data

    Returns:
        DataFrame with adjusted close prices
    """
    import yfinance as yf

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if len(df) > 0:
                # Handle different column name formats
                if 'Adj Close' in df.columns:
                    data[ticker] = df['Adj Close']
                elif 'Close' in df.columns:
                    data[ticker] = df['Close']
                elif isinstance(df.columns, pd.MultiIndex):
                    # Handle multi-level columns from yfinance
                    if ('Adj Close', ticker) in df.columns:
                        data[ticker] = df[('Adj Close', ticker)]
                    elif ('Close', ticker) in df.columns:
                        data[ticker] = df[('Close', ticker)]
                    else:
                        # Try flattening
                        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                        if 'Adj Close' in df.columns:
                            data[ticker] = df['Adj Close']
                        elif 'Close' in df.columns:
                            data[ticker] = df['Close']
                print(f"Fetched {len(df)} days for {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return pd.DataFrame(data)


def evaluate_forecast(actual: np.ndarray, forecast: np.ndarray, name: str = "Model"):
    """
    Evaluate forecast performance.

    Args:
        actual: Actual values
        forecast: Predicted values
        name: Model name for display

    Returns:
        Dictionary of metrics
    """
    # Align lengths
    n = min(len(actual), len(forecast))
    actual = actual[:n]
    forecast = forecast[:n]

    # Calculate metrics
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-8))) * 100

    # Directional accuracy
    actual_dir = np.sign(np.diff(actual))
    forecast_dir = np.sign(np.diff(forecast))
    direction_acc = np.mean(actual_dir == forecast_dir) * 100

    return {
        'name': name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'DirectionAcc': direction_acc
    }


def run_comparison_test(
    data: pd.DataFrame,
    target_col: str,
    exog_cols: list,
    train_size: int = 500,
    test_size: int = 30,
    n_paths: int = 500
):
    """
    Run comparison test across multiple models.

    Args:
        data: DataFrame with price data
        target_col: Column name for target series
        exog_cols: List of column names for exogenous variables
        train_size: Number of training observations
        test_size: Number of test observations
        n_paths: Number of Monte Carlo paths

    Returns:
        Results DataFrame
    """
    print("\n" + "="*70)
    print("FRACTAL FORECASTING WITH EXOGENOUS PREDICTORS")
    print("="*70)

    # Prepare data
    target = data[target_col].dropna().values
    dates = data.index.values

    # Prepare exogenous data
    exog_data = data[exog_cols].dropna()
    # Align with target
    common_idx = data[[target_col] + exog_cols].dropna().index
    target = data.loc[common_idx, target_col].values
    exog_df = data.loc[common_idx, exog_cols]
    dates = common_idx.values

    # Split into train/test
    train_end = len(target) - test_size
    train_prices = target[:train_end]
    test_prices = target[train_end:]
    train_dates = dates[:train_end]
    test_dates = dates[train_end:]
    train_exog = exog_df.iloc[:train_end]
    test_exog = exog_df.iloc[train_end:]

    print(f"\nTarget: {target_col}")
    print(f"Exogenous: {exog_cols}")
    print(f"Train size: {len(train_prices)}")
    print(f"Test size: {len(test_prices)}")
    print(f"Test period: {test_dates[0]} to {test_dates[-1]}")

    results = []

    # ========================================
    # 1. Fractal Forecaster (NO exogenous)
    # ========================================
    print("\n" + "-"*50)
    print("1. Fractal Forecaster (baseline, no exogenous)")
    print("-"*50)

    try:
        fc_base = ft.FractalForecaster(lookback=min(252, len(train_prices)), use_exogenous=False)
        fc_base.fit(train_prices, dates=train_dates)
        pred_base = fc_base.predict(n_steps=test_size, n_paths=n_paths)

        metrics_base = evaluate_forecast(
            test_prices,
            pred_base['weighted_forecast'],
            "Fractal (no exog)"
        )
        results.append(metrics_base)
        print(f"   RMSE: {metrics_base['RMSE']:.4f}")
        print(f"   Direction Accuracy: {metrics_base['DirectionAcc']:.1f}%")

        # Save for visualization
        fractal_base_result = pred_base
    except Exception as e:
        print(f"   ERROR: {e}")
        fractal_base_result = None

    # ========================================
    # 2. Fractal Forecaster WITH exogenous
    # ========================================
    print("\n" + "-"*50)
    print("2. Fractal Forecaster WITH exogenous predictors")
    print("-"*50)

    try:
        fc_exog = ft.FractalForecaster(
            lookback=min(252, len(train_prices)),
            use_exogenous=True,
            exog_max_lags=10,
            exog_min_correlation=0.05,
            exog_adjustment_strength=0.4
        )
        fc_exog.fit(train_prices, dates=train_dates, exogenous=train_exog)

        # Print exogenous summary
        exog_summary = fc_exog.get_exogenous_summary()
        if exog_summary:
            print("\n   Exogenous Analysis:")
            for name, info in exog_summary.get('variables', {}).items():
                status = "INCLUDED" if info['included'] else "excluded"
                print(f"     {name}: lag={info['best_lag']}, corr={info['correlation']:.3f} [{status}]")

        pred_exog = fc_exog.predict(n_steps=test_size, n_paths=n_paths)

        metrics_exog = evaluate_forecast(
            test_prices,
            pred_exog['weighted_forecast'],
            "Fractal (with exog)"
        )
        results.append(metrics_exog)
        print(f"\n   RMSE: {metrics_exog['RMSE']:.4f}")
        print(f"   Direction Accuracy: {metrics_exog['DirectionAcc']:.1f}%")

        # Calculate improvement
        if 'Fractal (no exog)' in [r['name'] for r in results]:
            base_rmse = next(r['RMSE'] for r in results if r['name'] == 'Fractal (no exog)')
            improvement = (base_rmse - metrics_exog['RMSE']) / base_rmse * 100
            print(f"   Improvement over baseline: {improvement:.1f}%")

        fractal_exog_result = pred_exog
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        fractal_exog_result = None

    # ========================================
    # 3. ARIMA baseline
    # ========================================
    print("\n" + "-"*50)
    print("3. ARIMA (baseline)")
    print("-"*50)

    try:
        arima = ARIMAForecaster()
        arima.fit(train_prices)
        pred_arima = arima.predict(n_steps=test_size)

        metrics_arima = evaluate_forecast(
            test_prices,
            pred_arima['forecast'],
            "ARIMA"
        )
        results.append(metrics_arima)
        print(f"   RMSE: {metrics_arima['RMSE']:.4f}")
        print(f"   Direction Accuracy: {metrics_arima['DirectionAcc']:.1f}%")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ========================================
    # 4. Exponential Smoothing
    # ========================================
    print("\n" + "-"*50)
    print("4. Exponential Smoothing")
    print("-"*50)

    try:
        ets = ETSForecaster()
        ets.fit(train_prices)
        pred_ets = ets.predict(n_steps=test_size)

        metrics_ets = evaluate_forecast(
            test_prices,
            pred_ets['forecast'],
            "ETS"
        )
        results.append(metrics_ets)
        print(f"   RMSE: {metrics_ets['RMSE']:.4f}")
        print(f"   Direction Accuracy: {metrics_ets['DirectionAcc']:.1f}%")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ========================================
    # 5. Random Forest (with exogenous)
    # ========================================
    print("\n" + "-"*50)
    print("5. Random Forest (with features)")
    print("-"*50)

    try:
        # Build features from exogenous variables
        rf = RandomForestForecaster(n_estimators=100)

        # Create lagged features
        X_train = train_exog.values[:-1]  # Lag by 1
        y_train = np.diff(np.log(train_prices))  # Returns

        # Align
        min_len = min(len(X_train), len(y_train))
        X_train = X_train[-min_len:]
        y_train = y_train[-min_len:]

        rf.fit(X_train, y_train)

        # Predict using last exog values (simple forecast)
        X_test = test_exog.values[:-1] if len(test_exog) > 1 else train_exog.values[-1:]
        pred_returns = rf.predict(X_test)

        # Convert returns to prices
        forecast_rf = train_prices[-1] * np.exp(np.cumsum(pred_returns[:test_size]))

        metrics_rf = evaluate_forecast(
            test_prices,
            forecast_rf,
            "Random Forest"
        )
        results.append(metrics_rf)
        print(f"   RMSE: {metrics_rf['RMSE']:.4f}")
        print(f"   Direction Accuracy: {metrics_rf['DirectionAcc']:.1f}%")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ========================================
    # 6. XGBoost (with exogenous)
    # ========================================
    print("\n" + "-"*50)
    print("6. XGBoost (with features)")
    print("-"*50)

    try:
        xgb = XGBoostForecaster(n_estimators=100)

        # Same feature setup as RF
        X_train = train_exog.values[:-1]
        y_train = np.diff(np.log(train_prices))

        min_len = min(len(X_train), len(y_train))
        X_train = X_train[-min_len:]
        y_train = y_train[-min_len:]

        xgb.fit(X_train, y_train)

        X_test = test_exog.values[:-1] if len(test_exog) > 1 else train_exog.values[-1:]
        pred_returns = xgb.predict(X_test)

        forecast_xgb = train_prices[-1] * np.exp(np.cumsum(pred_returns[:test_size]))

        metrics_xgb = evaluate_forecast(
            test_prices,
            forecast_xgb,
            "XGBoost"
        )
        results.append(metrics_xgb)
        print(f"   RMSE: {metrics_xgb['RMSE']:.4f}")
        print(f"   Direction Accuracy: {metrics_xgb['DirectionAcc']:.1f}%")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ========================================
    # Results Summary
    # ========================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')

    print("\n" + results_df.to_string(index=False))

    # ========================================
    # Create Interactive Visualization
    # ========================================
    print("\n" + "="*70)
    print("GENERATING INTERACTIVE VISUALIZATIONS")
    print("="*70)

    # Visualization 1: Fractal baseline
    if fractal_base_result is not None:
        print("\nCreating visualization: Fractal Baseline...")
        fig_base = ft.plot_forecast(
            train_prices[-100:],  # Show last 100 days of training
            fractal_base_result,
            dates=train_dates[-100:],
            title=f"{target_col} - Fractal Forecast (No Exogenous)",
            colorscale='Viridis',
            show_percentiles=True
        )
        fig_base.to_html("fractal_baseline_forecast.html")
        print("   Saved: fractal_baseline_forecast.html")

    # Visualization 2: Fractal with exogenous
    if fractal_exog_result is not None:
        print("\nCreating visualization: Fractal with Exogenous...")
        fig_exog = ft.plot_forecast(
            train_prices[-100:],
            fractal_exog_result,
            dates=train_dates[-100:],
            title=f"{target_col} - Fractal Forecast (With Exogenous: {', '.join(exog_cols)})",
            colorscale='Plasma',
            show_percentiles=True
        )
        fig_exog.to_html("fractal_exogenous_forecast.html")
        print("   Saved: fractal_exogenous_forecast.html")

    # Visualization 3: Comparison chart
    if fractal_base_result is not None and fractal_exog_result is not None:
        print("\nCreating visualization: Comparison Chart...")
        from wrchart import MultiPanelChart
        from wrchart.multipanel import LinePanel

        # Prepare data
        hist_prices = train_prices[-50:].tolist()
        hist_x = list(range(len(hist_prices)))

        test_x = list(range(len(hist_prices), len(hist_prices) + len(test_prices)))
        actual_prices = test_prices.tolist()

        base_forecast = fractal_base_result['weighted_forecast'][:len(test_prices)].tolist()
        exog_forecast = fractal_exog_result['weighted_forecast'][:len(test_prices)].tolist()

        # Combine all data for multi-line plot
        all_x = hist_x + test_x
        n_hist = len(hist_prices)
        n_test = len(test_prices)

        # Historical line (padded with None for forecast region)
        historical_line = hist_prices + [None] * n_test

        # Actual line (padded with None for historical region, starts from last historical)
        actual_line = [None] * (n_hist - 1) + [hist_prices[-1]] + actual_prices

        # Baseline forecast line
        baseline_line = [None] * (n_hist - 1) + [hist_prices[-1]] + base_forecast

        # Exogenous forecast line
        exog_line = [None] * (n_hist - 1) + [hist_prices[-1]] + exog_forecast

        chart = MultiPanelChart(
            rows=1,
            cols=1,
            width=1000,
            height=600,
            title=f"{target_col} - Model Comparison",
            theme="dark",
        )

        chart.add_panel(LinePanel(
            title='',
            x_data=all_x,
            y_data=[historical_line, actual_line, baseline_line, exog_line],
            colors=['white', '#4CAF50', '#2196F3', '#F44336'],
            line_widths=[2, 3, 2, 2],
            labels=['Historical', 'Actual', 'Fractal (baseline)', 'Fractal (with exog)'],
            row=0,
            col=0,
        ))

        chart.to_html("model_comparison.html")
        print("   Saved: model_comparison.html")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    return results_df


def main():
    """Main function to run the comparison test."""

    print("\n" + "#"*70)
    print("# FRACTAL FORECASTING WITH EXOGENOUS PREDICTORS")
    print("# Comparison Test Suite")
    print("#"*70)

    # Fetch data
    print("\nFetching market data...")
    tickers = ['SPY', '^VIX', 'TLT', 'GLD', 'DX-Y.NYB']  # S&P 500, VIX, Bonds, Gold, Dollar Index
    data = fetch_test_data(tickers, start_date='2019-01-01')

    if data.empty:
        print("No data fetched. Using synthetic data...")
        # Create synthetic data
        np.random.seed(42)
        n = 1000
        dates = pd.date_range(start='2020-01-01', periods=n, freq='B')

        # Generate correlated series
        noise = np.random.randn(n, 4)
        cov_matrix = np.array([
            [1.0, -0.3, 0.2, 0.1],
            [-0.3, 1.0, -0.1, 0.05],
            [0.2, -0.1, 1.0, 0.3],
            [0.1, 0.05, 0.3, 1.0]
        ])
        correlated = noise @ np.linalg.cholesky(cov_matrix).T

        data = pd.DataFrame({
            'SPY': 300 * np.exp(np.cumsum(0.0005 + 0.01 * correlated[:, 0])),
            '^VIX': 20 * np.exp(np.cumsum(0.02 * correlated[:, 1])),
            'TLT': 140 * np.exp(np.cumsum(0.0003 + 0.005 * correlated[:, 2])),
            'GLD': 150 * np.exp(np.cumsum(0.0002 + 0.008 * correlated[:, 3]))
        }, index=dates)

    # Clean column names (remove ^ prefix if present)
    data.columns = [c.replace('^', '') for c in data.columns]

    # Run comparison test
    results = run_comparison_test(
        data=data,
        target_col='SPY',
        exog_cols=['VIX', 'TLT', 'GLD'] if 'VIX' in data.columns else list(data.columns[1:]),
        train_size=500,
        test_size=30,
        n_paths=500
    )

    print("\nFinal Results:")
    print(results)


if __name__ == "__main__":
    main()
