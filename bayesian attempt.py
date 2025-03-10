import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta
from collections import deque
import warnings
import json
from functools import partial
import joblib

# Try importing scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize, forest_minimize, dummy_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.plots import plot_convergence, plot_objective, plot_evaluations
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize not available. Install with: pip install scikit-optimize")
    print("Falling back to random search")

warnings.filterwarnings('ignore')

def load_candle_data(directory, timeframe, start_date=None, end_date=None):
    """Load candle data from CSV with optional date filtering"""
    file_path = os.path.join(directory, f'GBPJPY_{timeframe}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Apply date filtering if provided
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
        
    return df.sort_values('timestamp').reset_index(drop=True)

def calculate_weighted_close(high, low, close, kama_style=True):
    """Calculate weighted close price"""
    if kama_style:
        # (High + Low + Close*2) / 4
        return (high + low + 2 * close) / 4
    else:
        # (High + Low + Close) / 3 - HLC
        return (high + low + close) / 3

# ========== CUSTOM TREND CHANGE INDICATOR ==========
def calculate_trend_change_indicator(df, params):
    """
    Calculate a binary trend change indicator (1 = signal, 0 = no signal)
    
    This is where you'll implement your custom indicator logic.
    
    Args:
        df: DataFrame with price and indicator data
        params: Dictionary of parameters for calculation
        
    Returns:
        Series with binary signals (1 or 0)
    """
    # Create a copy of the dataframe to avoid modifying the original
    result = pd.Series(0, index=df.index)
    
    # Extract parameters
    lookback = params.get('trend_change_lookback', 10)
    threshold = params.get('trend_change_threshold', 0.0005)
    indicator1 = params.get('trend_change_indicator1', 'sma_5m')
    indicator2 = params.get('trend_change_indicator2', 'sma_15m')
    
    # =========================================================
    # MODIFY THIS SECTION WITH YOUR CUSTOM INDICATOR FORMULA
    # =========================================================
    
    # This is just an example formula - replace with your own logic
    # Current implementation: Signal when indicator1 crosses over indicator2
    if indicator1 in df.columns and indicator2 in df.columns:
        # Calculate the difference between indicators
        df['diff'] = df[indicator1] - df[indicator2]
        
        # Look for crossovers (sign changes in the difference)
        df['prev_diff'] = df['diff'].shift(1)
        
        # Find where the sign changes from negative to positive (bullish crossover)
        bullish = (df['prev_diff'] < 0) & (df['diff'] > 0)
        
        # Find where the sign changes from positive to negative (bearish crossover)
        bearish = (df['prev_diff'] > 0) & (df['diff'] < 0)
        
        # Set signal to 1 where a crossover occurs
        result[bullish] = 1  # Bullish signal
        result[bearish] = 1  # Bearish signal
    
    # =========================================================
    # END OF CUSTOM FORMULA SECTION
    # =========================================================
    
    return result

def calculate_indicators(candles_5m, candles_15m, candles_30m, params):
    """
    Calculate selected indicators with configurable parameters
    
    Args:
        candles_5m: DataFrame with 5M candles
        candles_15m: DataFrame with 15M candles
        candles_30m: DataFrame with 30M candles
        params: Dictionary of parameters for indicator calculation
        
    Returns:
        DataFrame with calculated indicators
    """
    print(f"Calculating indicators with parameters: {params}")
    
    # Extract parameters
    ker_period = params.get('ker_period', 10)
    kama_period = params.get('kama_period', 10)
    fast_period = params.get('fast_period', 2)
    slow_period = params.get('slow_period', 30)
    
    # Create results dataframe starting with 5M data
    results = candles_5m.copy()
    
    # Add timeframe period information
    results['period_15m'] = (results['timestamp'].dt.minute % 15) // 5
    results['period_30m'] = (results['timestamp'].dt.minute % 30) // 5
    
    # Determine which indicators to calculate
    calculate_5m = params.get('calculate_5m', True)
    calculate_15m = params.get('calculate_15m', True)
    calculate_30m = params.get('calculate_30m', True)
    
    # Initialize result columns for selected timeframes
    timeframes = []
    if calculate_5m:
        timeframes.append('5m')
    if calculate_15m:
        timeframes.append('15m')
    if calculate_30m:
        timeframes.append('30m')
    
    for tf in timeframes:
        results[f'sma_{tf}'] = np.nan
        results[f'ker_{tf}'] = np.nan
        results[f'kama_{tf}'] = np.nan
    
    # Setup initial KAMA values
    last_kama = {tf: None for tf in timeframes}
    
    # Window sizes - can be customized via parameters
    window_sizes = {
        '5m': params.get('sma_5m_period', 12),    # Default: 1 hour
        '15m': params.get('sma_15m_period', 20),  # Default: 5 hours
        '30m': params.get('sma_30m_period', 30)   # Default: 15 hours
    }
    
    # Create lookup maps for higher timeframes
    candles_15m_map = {}
    for _, row in candles_15m.iterrows():
        candles_15m_map[row['timestamp']] = row.to_dict()
    
    candles_30m_map = {}
    for _, row in candles_30m.iterrows():
        candles_30m_map[row['timestamp']] = row.to_dict()
    
    # Create windows for rolling calculations - make sure they're large enough 
    # for the largest requested period
    max_window_size = max(max(window_sizes.values()), ker_period, kama_period)
    
    windows = {tf: deque(maxlen=max_window_size + 10) for tf in timeframes}
    
    # Sort by timestamp to ensure chronological processing
    results = results.sort_values('timestamp').reset_index(drop=True)
    
    # For each 5M candle, process indicators
    for idx, row in results.iterrows():
        current_time = row['timestamp']
        
        # Progress indicator every 5000 rows
        if idx % 5000 == 0 and idx > 0:
            print(f"Processed {idx}/{len(results)} rows ({idx/len(results)*100:.1f}%)")
        
        # Add current 5M candle to window if 5M timeframe is selected
        if calculate_5m:
            windows['5m'].append(row.to_dict())
        
        # Find corresponding 15M candle if 15M timeframe is selected
        if calculate_15m:
            current_15m_time = current_time.floor('15min')
            if current_15m_time in candles_15m_map:
                # If this is a new 15M candle we haven't seen yet, add it to window
                if len(windows['15m']) == 0 or windows['15m'][-1]['timestamp'] != current_15m_time:
                    windows['15m'].append(candles_15m_map[current_15m_time])
        
        # Find corresponding 30M candle if 30M timeframe is selected
        if calculate_30m:
            current_30m_time = current_time.floor('30min')
            if current_30m_time in candles_30m_map:
                # If this is a new 30M candle we haven't seen yet, add it to window
                if len(windows['30m']) == 0 or windows['30m'][-1]['timestamp'] != current_30m_time:
                    windows['30m'].append(candles_30m_map[current_30m_time])
        
        # Calculate indicators for each selected timeframe
        for tf in timeframes:
            window = windows[tf]
            if len(window) == 0:
                continue
                
            # Calculate SMA if we have enough data
            if len(window) >= window_sizes[tf]:
                sma = sum(candle['close'] for candle in list(window)[-window_sizes[tf]:]) / window_sizes[tf]
                results.loc[idx, f'sma_{tf}'] = round(sma, 5)
            
            # Calculate KER if we have enough data - using KER period
            if len(window) >= ker_period:
                # Get candles for KER calculation
                ker_candles = list(window)[-ker_period:]
                
                # Calculate weighted closes for KER (HLC)
                wcloses = [calculate_weighted_close(c['high'], c['low'], c['close'], kama_style=False) 
                          for c in ker_candles]
                
                # Calculate price change (direction)
                price_change = abs(wcloses[-1] - wcloses[0])
                
                # Calculate volatility (path)
                volatility = sum(abs(wcloses[i] - wcloses[i-1]) for i in range(1, len(wcloses)))
                
                # Calculate ER
                if volatility > 0:
                    er_value = price_change / volatility
                else:
                    er_value = 0
                
                results.loc[idx, f'ker_{tf}'] = round(er_value, 5)
                
                # Calculate KAMA - only if we have KER value
                current_candle = window[-1]
                current_price = calculate_weighted_close(
                    current_candle['high'], 
                    current_candle['low'], 
                    current_candle['close'], 
                    kama_style=True
                )
                
                # Initialize KAMA if needed - using KAMA period
                if last_kama[tf] is None:
                    if len(window) >= kama_period:
                        # Initialize with average of last kama_period candles
                        kama_candles = list(window)[-kama_period:]
                        weighted_closes = [calculate_weighted_close(
                            c['high'], c['low'], c['close'], kama_style=True) 
                            for c in kama_candles]
                        last_kama[tf] = sum(weighted_closes) / len(weighted_closes)
                    else:
                        # Not enough data yet
                        continue
                
                # Calculate smoothing constant - using fast and slow period parameters
                fast_sc = 2.0 / (fast_period + 1.0)
                slow_sc = 2.0 / (slow_period + 1.0)
                sc = (er_value * (fast_sc - slow_sc) + slow_sc) ** 2
                
                # Calculate KAMA
                kama_value = last_kama[tf] + sc * (current_price - last_kama[tf])
                results.loc[idx, f'kama_{tf}'] = round(kama_value, 5)
                
                # Update last KAMA
                last_kama[tf] = kama_value
    
    print("Indicator calculation complete")
    
    # Create combined indicators only if all required timeframes are available
    if all(tf in timeframes for tf in ['5m', '15m', '30m']):
        print("Creating combined indicators...")
        
        # Create only the requested combined indicators
        if params.get('calculate_tska', False):
            # Trend-Strength-Adjusted KAMA
            results['TSKA'] = results['kama_15m'] * (1 + results['ker_5m'] - results['ker_30m'])
        
        if params.get('calculate_afe', False):
            # Adaptive Fractal Efficiency
            results['AFE'] = results['ker_5m'] * (results['kama_5m'] / results['kama_15m']) * np.sign(results['kama_15m'] - results['kama_30m'])
        
        if params.get('calculate_msm', False):
            # Calculate ROC first
            for tf in timeframes:
                results[f'kama_{tf}_roc'] = results[f'kama_{tf}'].pct_change(3) * 100
                
            # Multi-Scale Momentum
            results['MSM'] = (
                (results['kama_5m_roc'] * results['ker_5m'] + 
                 results['kama_15m_roc'] * results['ker_15m'] + 
                 results['kama_30m_roc'] * results['ker_30m']) / 
                (results['ker_5m'] + results['ker_15m'] + results['ker_30m'])
            )
        
        if params.get('calculate_hier_momentum', False):
            # Calculate ROC if not already done
            if 'kama_5m_roc' not in results.columns:
                for tf in timeframes:
                    results[f'kama_{tf}_roc'] = results[f'kama_{tf}'].pct_change(3) * 100
                    
            # Hierarchical Momentum
            results['hier_momentum'] = (
                np.sign(results['kama_5m_roc']) * 
                np.sign(results['kama_15m_roc']) * 
                np.sign(results['kama_30m_roc']) * 
                results[['kama_5m_roc', 'kama_15m_roc', 'kama_30m_roc']].abs().max(axis=1)
            )
        
        if params.get('calculate_ata', False):
            # Adaptive Timeframe Allocation
            results['ATA'] = (
                (results['kama_5m'] * results['ker_5m'] + 
                 results['kama_15m'] * results['ker_15m'] + 
                 results['kama_30m'] * results['ker_30m']) /
                (results['ker_5m'] + results['ker_15m'] + results['ker_30m'])
            )
    
    # Calculate additional indicators
    # Rate of change
    for tf in timeframes:
        if f'sma_{tf}' in results.columns:
            results[f'sma_{tf}_roc_1'] = results[f'sma_{tf}'].pct_change(1) * 100
            results[f'sma_{tf}_roc_3'] = results[f'sma_{tf}'].pct_change(3) * 100
            
        if f'kama_{tf}' in results.columns:
            results[f'kama_{tf}_roc_1'] = results[f'kama_{tf}'].pct_change(1) * 100
            results[f'kama_{tf}_roc_3'] = results[f'kama_{tf}'].pct_change(3) * 100
    
    # Calculate the trend change indicator (binary signal)
    results['trend_change_signal'] = calculate_trend_change_indicator(results, params)
    
    # Remove NaN values
    results = results.dropna(subset=[col for col in results.columns if col not in ['timestamp', 'period_15m', 'period_30m']], how='all')
    
    return results

def calculate_future_values(df, params):
    """
    Calculate future SMA values for prediction targets
    
    Args:
        df: DataFrame with indicators
        params: Parameters for future calculation
        
    Returns:
        DataFrame with added future columns
    """
    results = df.copy()
    
    # Get future periods to calculate
    future_periods = params.get('future_periods', [3, 5, 10])
    
    # Get which SMA to use for future calculation
    future_sma_tf = params.get('future_sma_tf', '5m')
    future_sma_period = params.get('future_sma_period', 3)
    
    # Calculate future SMA
    for period in future_periods:
        # Calculate future price
        results[f'future_price_{period}'] = results['price'].shift(-period)
        
        # Calculate future SMA if enough data
        if len(results) > period + future_sma_period:
            # Create a temporary shifted dataframe for calculating future SMA
            temp_df = results[['timestamp', 'price']].copy()
            temp_df['future_price'] = temp_df['price'].shift(-period)
            
            # Calculate rolling SMA on future prices
            future_sma = temp_df['future_price'].rolling(window=future_sma_period).mean()
            
            # Add to results
            results[f'future_sma_{period}_{future_sma_period}'] = future_sma
            
            # Calculate direction (up/down) compared to current price or SMA
            results[f'future_direction_{period}_{future_sma_period}'] = np.sign(
                results[f'future_sma_{period}_{future_sma_period}'] - results['price']
            )
            
            # Calculate direction compared to current SMA
            current_sma_col = f'sma_{future_sma_tf}'
            if current_sma_col in results.columns:
                results[f'future_sma_vs_current_sma_{period}_{future_sma_period}'] = np.sign(
                    results[f'future_sma_{period}_{future_sma_period}'] - results[current_sma_col]
                )
    
    return results

def evaluate_binary_signal_performance(df, params):
    """
    Evaluate how well the binary trend change indicator predicts future price/SMA direction
    
    Args:
        df: DataFrame with indicators and future values
        params: Parameters for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract parameters
    future_period = params.get('future_period', 5)
    future_sma_period = params.get('future_sma_period', 3)
    comparison_mode = params.get('comparison_mode', 'sma_vs_price')  # 'sma_vs_price' or 'sma_vs_sma'
    signal_column = params.get('signal_column', 'trend_change_signal')
    direction_after_signal = params.get('direction_after_signal', 'up')  # 'up' or 'down' or 'auto'
    
    # Check if required columns exist
    if signal_column not in df.columns:
        print(f"Warning: Signal column {signal_column} not found in data")
        return {'accuracy': 0, 'signal_count': 0}
    
    if comparison_mode == 'sma_vs_price':
        future_col = f'future_direction_{future_period}_{future_sma_period}'
    else:  # 'sma_vs_sma'
        future_col = f'future_sma_vs_current_sma_{future_period}_{future_sma_period}'
    
    if future_col not in df.columns:
        print(f"Warning: Future column {future_col} not found in data")
        return {'accuracy': 0, 'signal_count': 0}
    
    # Find rows where signal is 1
    signal_rows = df[df[signal_column] == 1]
    
    if len(signal_rows) == 0:
        print(f"Warning: No signals found in data")
        return {'accuracy': 0, 'signal_count': 0}
    
    # Get future directions after signals
    signal_future_directions = signal_rows[future_col]
    
    # Handle auto-detection of best direction
    if direction_after_signal == 'auto':
        # Count positive and negative directions
        positive_count = (signal_future_directions > 0).sum()
        negative_count = (signal_future_directions < 0).sum()
        
        # Choose the direction that occurs more frequently
        if positive_count >= negative_count:
            expected_direction = 1  # up
        else:
            expected_direction = -1  # down
    else:
        # Use the specified direction
        expected_direction = 1 if direction_after_signal == 'up' else -1
    
    # Calculate accuracy
    correct_predictions = (signal_future_directions == expected_direction).sum()
    total_signals = len(signal_rows)
    
    accuracy = correct_predictions / total_signals if total_signals > 0 else 0
    
    # Calculate additional metrics
    signal_ratio = total_signals / len(df)
    
    # Calculate price change after signals
    price_changes = []
    future_sma_changes = []
    
    for idx in signal_rows.index:
        if idx + future_period < len(df):
            price_change = df.loc[idx + future_period, 'price'] - df.loc[idx, 'price']
            price_changes.append(price_change)
            
            future_sma_col = f'future_sma_{future_period}_{future_sma_period}'
            if future_sma_col in df.columns:
                if not pd.isna(df.loc[idx, future_sma_col]) and not pd.isna(df.loc[idx, 'price']):
                    future_sma_change = df.loc[idx, future_sma_col] - df.loc[idx, 'price']
                    future_sma_changes.append(future_sma_change)
    
    avg_price_change = np.mean(price_changes) if price_changes else 0
    avg_future_sma_change = np.mean(future_sma_changes) if future_sma_changes else 0
    
    return {
        'accuracy': accuracy,
        'signal_count': total_signals,
        'signal_ratio': signal_ratio,
        'expected_direction': expected_direction,
        'avg_price_change': avg_price_change,
        'avg_future_sma_change': avg_future_sma_change
    }

def optimize_binary_signal_prediction(data_directory, output_directory, parameter_space=None, 
                                  n_calls=100, train_period=None, test_period=None, random_state=42):
    """
    Perform Bayesian optimization for binary signal prediction
    
    Args:
        data_directory: Directory containing candle data files
        output_directory: Directory to save optimization results
        parameter_space: Dictionary defining parameter search space (if None, use default)
        n_calls: Number of optimization iterations
        train_period: Tuple of (start_date, end_date) for training
        test_period: Tuple of (start_date, end_date) for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with optimization results
    """
    # Check if scikit-optimize is available
    if not SKOPT_AVAILABLE:
        print("Warning: scikit-optimize not available. Falling back to random search.")
        print("Install scikit-optimize with: pip install scikit-optimize")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Define train and test periods
    if train_period is None:
        train_start = datetime(2023, 1, 1)
        train_end = datetime(2023, 12, 31)
    else:
        train_start, train_end = train_period
    
    if test_period is None:
        test_start = datetime(2024, 1, 1)
        test_end = datetime(2024, 3, 1)
    else:
        test_start, test_end = test_period
    
    print(f"Train period: {train_start} to {train_end}")
    print(f"Test period: {test_start} to {test_end}")
    
    # Load data for the entire period
    all_start = min(train_start, test_start)
    all_end = max(train_end, test_end)
    
    print(f"Loading data from {all_start} to {all_end}...")
    
    try:
        candles_5m = load_candle_data(data_directory, '5M', all_start, all_end)
        candles_15m = load_candle_data(data_directory, '15M', all_start, all_end)
        candles_30m = load_candle_data(data_directory, '30M', all_start, all_end)
        
        print(f"Loaded {len(candles_5m)} 5M candles")
        print(f"Loaded {len(candles_15m)} 15M candles")
        print(f"Loaded {len(candles_30m)} 30M candles")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Define default parameter space if not provided
    if parameter_space is None:
        parameter_space = {
            # Indicator calculation parameters
            'ker_period': (5, 20),
            'kama_period': (5, 20),
            'fast_period': (2, 5),
            'slow_period': (20, 40),
            
            # SMA periods
            'sma_5m_period': (3, 20),
            'sma_15m_period': (3, 20),
            'sma_30m_period': (3, 20),
            
            # Timeframe selection
            'calculate_5m': [True],
            'calculate_15m': [True],
            'calculate_30m': [True],
            
            # Composite indicator selection
            'calculate_tska': [True, False],
            'calculate_afe': [True, False],
            'calculate_msm': [True, False],
            'calculate_hier_momentum': [True, False],
            'calculate_ata': [True, False],
            
            # Trend change indicator parameters
            'trend_change_lookback': (5, 20),
            'trend_change_threshold': (0.0001, 0.001),
            'trend_change_indicator1': ['sma_5m', 'sma_15m', 'sma_30m', 'kama_5m', 'kama_15m', 'kama_30m'],
            'trend_change_indicator2': ['sma_5m', 'sma_15m', 'sma_30m', 'kama_5m', 'kama_15m', 'kama_30m'],
            
            # Binary signal evaluation parameters
            'future_period': [3, 5, 10, 20],
            'future_sma_period': [3, 5, 10],
            'comparison_mode': ['sma_vs_price', 'sma_vs_sma'],
            'direction_after_signal': ['up', 'down', 'auto']
        }
    
    # Convert parameter space to scikit-optimize format
    space = []
    for param_name, param_range in parameter_space.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Numeric parameter
            if all(isinstance(x, int) for x in param_range):
                space.append(Integer(param_range[0], param_range[1], name=param_name))
            else:
                space.append(Real(param_range[0], param_range[1], name=param_name))
        elif isinstance(param_range, list):
            # Categorical parameter
            space.append(Categorical(param_range, name=param_name))
        else:
            print(f"Warning: Invalid parameter range for {param_name}: {param_range}")
    
    # Define objective function
    def objective(params_list):
        # Convert list of parameter values to dictionary
        params = {dim.name: value for dim, value in zip(space, params_list)}
        
        # Extract training data
        train_5m = candles_5m[(candles_5m['timestamp'] >= train_start) & 
                              (candles_5m['timestamp'] <= train_end)]
        
        train_15m = candles_15m[(candles_15m['timestamp'] >= train_start) & 
                               (candles_15m['timestamp'] <= train_end)]
        
        train_30m = candles_30m[(candles_30m['timestamp'] >= train_start) & 
                               (candles_30m['timestamp'] <= train_end)]
        
        # Calculate indicators on training data
        indicators = calculate_indicators(train_5m, train_15m, train_30m, params)
        
        # Calculate future values for prediction targets
        future_params = {
            'future_periods': [params['future_period']],
            'future_sma_tf': '5m',
            'future_sma_period': params['future_sma_period']
        }
        full_data = calculate_future_values(indicators, future_params)
        
        # Evaluate binary signal performance
        evaluation = evaluate_binary_signal_performance(full_data, params)
        
        # Use negative accuracy as objective (to maximize)
        score = -evaluation['accuracy']
        
        print(f"Parameters: {params}")
        print(f"Signal Count: {evaluation['signal_count']}")
        print(f"Accuracy: {evaluation['accuracy']:.4f} (higher is better)")
        print(f"Expected Direction: {'Up' if evaluation['expected_direction'] > 0 else 'Down'}")
        print(f"Avg Price Change: {evaluation['avg_price_change']:.6f}")
        print(f"Avg Future SMA Change: {evaluation['avg_future_sma_change']:.6f}")
        print("-" * 50)
        
        # Penalize if there are too few signals
        if evaluation['signal_count'] < 10:
            score = 0  # Heavy penalty for too few signals
        
        return score
    
    # Perform optimization
    start_time = time.time()
    
    if SKOPT_AVAILABLE:
        # Use scikit-optimize for Bayesian optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=random_state,
            verbose=True
        )
        
        # Get best parameters
        best_params = {dim.name: value for dim, value in zip(space, result.x)}
        best_score = -result.fun
        
        # Save optimization results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_directory, f"optimization_result_{timestamp}.pkl")
        joblib.dump(result, result_file)
        print(f"Optimization result saved to {result_file}")
        
        # Create plots
        plt.figure(figsize=(10, 6))
        plot_convergence(result)
        plt.savefig(os.path.join(output_directory, f"convergence_{timestamp}.png"))
        
        plt.figure(figsize=(12, 8))
        plot_objective(result)
        plt.savefig(os.path.join(output_directory, f"objective_{timestamp}.png"))
        
        plt.figure(figsize=(12, 8))
        plot_evaluations(result)
        plt.savefig(os.path.join(output_directory, f"evaluations_{timestamp}.png"))
    else:
        # Fallback to random search
        best_score = -float('inf')
        best_params = None
        results = []
        
        for i in range(n_calls):
            print(f"Iteration {i+1}/{n_calls}")
            
            # Generate random parameters
            params_list = []
            for dim in space:
                if isinstance(dim, Integer):
                    params_list.append(np.random.randint(dim.low, dim.high + 1))
                elif isinstance(dim, Real):
                    params_list.append(np.random.uniform(dim.low, dim.high))
                elif isinstance(dim, Categorical):
                    params_list.append(np.random.choice(dim.categories))
            
            # Evaluate objective
            score = objective(params_list)
            results.append((params_list, score))
            
            # Update best score
            if -score > best_score:
                best_score = -score
                best_params = {dim.name: value for dim, value in zip(space, params_list)}
        
        # Save random search results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_directory, f"random_search_result_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump({'best_params': best_params, 'best_score': best_score}, f, indent=2)
        print(f"Random search result saved to {result_file}")
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    
    # Evaluate on test set
    print("Evaluating best parameters on test set...")
    
    test_5m = candles_5m[(candles_5m['timestamp'] >= test_start) & 
                         (candles_5m['timestamp'] <= test_end)]
    
    test_15m = candles_15m[(candles_15m['timestamp'] >= test_start) & 
                          (candles_15m['timestamp'] <= test_end)]
    
    test_30m = candles_30m[(candles_30m['timestamp'] >= test_start) & 
                          (candles_30m['timestamp'] <= test_end)]
    
    # Calculate indicators on test data
    test_indicators = calculate_indicators(test_5m, test_15m, test_30m, best_params)
    
    # Calculate future values for prediction targets
    future_params = {
        'future_periods': [best_params['future_period']],
        'future_sma_tf': '5m',
        'future_sma_period': best_params['future_sma_period']
    }
    test_data = calculate_future_values(test_indicators, future_params)
    
    # Evaluate binary signal performance
    test_evaluation = evaluate_binary_signal_performance(test_data, best_params)
    
    print("Test set evaluation:")
    print(f"Signal Count: {test_evaluation['signal_count']}")
    print(f"Accuracy: {test_evaluation['accuracy']:.4f}")
    print(f"Expected Direction: {'Up' if test_evaluation['expected_direction'] > 0 else 'Down'}")
    print(f"Avg Price Change: {test_evaluation['avg_price_change']:.6f}")
    print(f"Avg Future SMA Change: {test_evaluation['avg_future_sma_change']:.6f}")
    
    # Create visualization of signals and performance
    if test_evaluation['signal_count'] > 0:
        # Extract signal rows
        signal_rows = test_data[test_data['trend_change_signal'] == 1]
        
        # Plot price and signals
        plt.figure(figsize=(14, 10))
        
        # Plot price
        ax1 = plt.subplot(211)
        ax1.plot(test_data['timestamp'], test_data['price'], label='Price')
        
        # Determine which comparison mode and future column to use
        if best_params['comparison_mode'] == 'sma_vs_price':
            future_col = f'future_direction_{best_params["future_period"]}_{best_params["future_sma_period"]}'
        else:
            future_col = f'future_sma_vs_current_sma_{best_params["future_period"]}_{best_params["future_sma_period"]}'
        
        # Filter signals by expected direction
        correct_signals = signal_rows[signal_rows[future_col] == test_evaluation['expected_direction']]
        incorrect_signals = signal_rows[signal_rows[future_col] != test_evaluation['expected_direction']]
        
        # Plot correct signals
        ax1.scatter(correct_signals['timestamp'], 
                  correct_signals['price'], 
                  color='green', marker='^', s=100, label='Correct Signal')
        
        # Plot incorrect signals
        ax1.scatter(incorrect_signals['timestamp'], 
                  incorrect_signals['price'], 
                  color='red', marker='v', s=100, label='Incorrect Signal')
        
        ax1.set_title('Price and Signal Performance')
        ax1.legend()
        
        # Plot indicators used in the trend change signal
        ax2 = plt.subplot(212, sharex=ax1)
        
        indicator1 = best_params['trend_change_indicator1']
        indicator2 = best_params['trend_change_indicator2']
        
        if indicator1 in test_data.columns and indicator2 in test_data.columns:
            ax2.plot(test_data['timestamp'], test_data[indicator1], label=indicator1)
            ax2.plot(test_data['timestamp'], test_data[indicator2], label=indicator2)
            
            # Plot crossover points
            crossover_points = test_data[test_data['trend_change_signal'] == 1]
            
            for idx, row in crossover_points.iterrows():
                # Draw vertical line at signal point
                ax2.axvline(x=row['timestamp'], color='purple', linestyle='--', alpha=0.5)
        
        ax2.set_title(f'Indicators: {indicator1} vs {indicator2}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f"signal_performance_{timestamp}.png"))
        
        # Save detailed signal data
        signal_file = os.path.join(output_directory, f"signal_details_{timestamp}.csv")
        signal_rows.to_csv(signal_file, index=False)
        print(f"Signal details saved to {signal_file}")
    
    # Save test results
    test_result_file = os.path.join(output_directory, f"test_results_{timestamp}.json")
    with open(test_result_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score,
            'test_evaluation': test_evaluation,
            'optimization_time': optimization_time
        }, f, indent=2)
    print(f"Test results saved to {test_result_file}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'test_evaluation': test_evaluation
    }

def main():
    """Main function for binary signal optimization"""
    # Set directories
    data_directory = r'D:\\'  # Update this path to your data directory
    output_directory = os.path.join(data_directory, 'signal_results')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Set optimization parameters
    # Define train and test periods
    train_period = (datetime(2023, 1, 1), datetime(2023, 12, 31))  # Use 2023 for training
    test_period = (datetime(2024, 1, 1), datetime(2024, 3, 1))     # Test on 2024 data
    
    # Define parameter space for optimization
    parameter_space = {
        # Indicator calculation parameters
        'ker_period': (5, 15),
        'kama_period': (5, 15),
        'fast_period': (2, 5),
        'slow_period': (20, 40),
        
        # SMA periods to test
        'sma_5m_period': (3, 20),
        'sma_15m_period': (3, 20),
        'sma_30m_period': (3, 20),
        
        # Timeframe selection (which timeframes to calculate)
        'calculate_5m': [True],
        'calculate_15m': [True],
        'calculate_30m': [True],
        
        # Composite indicator selection (which advanced indicators to calculate)
        'calculate_tska': [True],  # Trend-Strength-Adjusted KAMA
        'calculate_afe': [True],   # Adaptive Fractal Efficiency
        'calculate_msm': [True],   # Multi-Scale Momentum
        'calculate_hier_momentum': [True],  # Hierarchical Momentum
        'calculate_ata': [True],   # Adaptive Timeframe Allocation
        
        # Trend change indicator parameters
        # ===== CUSTOMIZE THESE PARAMETERS FOR YOUR CUSTOM INDICATOR =====
        'trend_change_lookback': (5, 20),  # Lookback period for the indicator
        'trend_change_threshold': (0.0001, 0.001),  # Threshold for signal generation
        'trend_change_indicator1': [  # First indicator for comparison
            'sma_5m', 'sma_15m', 'sma_30m',
            'kama_5m', 'kama_15m', 'kama_30m',
            'ker_5m', 'ker_15m', 'ker_30m'
        ],
        'trend_change_indicator2': [  # Second indicator for comparison
            'sma_5m', 'sma_15m', 'sma_30m',
            'kama_5m', 'kama_15m', 'kama_30m', 
            'ker_5m', 'ker_15m', 'ker_30m'
        ],
        
        # Future prediction parameters
        'future_period': [3, 6, 12, 24],  # 15, 30, 60, 120 minutes ahead
        'future_sma_period': [3, 5, 10],  # SMA periods for future calculation
        'comparison_mode': ['sma_vs_price', 'sma_vs_sma'],  # How to compare
        'direction_after_signal': ['up', 'down', 'auto']  # Expected direction after signal
    }
    
    # Perform optimization
    result = optimize_binary_signal_prediction(
        data_directory=data_directory,
        output_directory=output_directory,
        parameter_space=parameter_space,
        n_calls=50,  # Number of optimization iterations (increase for better results)
        train_period=train_period,
        test_period=test_period,
        random_state=42
    )
    
    if result is not None:
        print("Optimization completed successfully!")
        print("Best parameters:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
        print(f"Best accuracy: {result['best_score']:.4f}")
        print("Test performance:")
        for metric, value in result['test_evaluation'].items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("Optimization failed. Check the error messages above.")

if __name__ == "__main__":
    main()
