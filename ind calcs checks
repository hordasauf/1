import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

def load_candle_data(directory, timeframe, start_date=None, end_date=None):
    """
    Load candle data from CSV with optional date filtering
    
    Args:
        directory: Directory containing data files
        timeframe: Timeframe string (e.g., '5M', '15M', '30M')
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        DataFrame with candle data
    """
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

def calculate_indicators_for_dataset(candles_5m, candles_15m, candles_30m, 
                               ker_period=10, kama_period=10, 
                               fast_period=2, slow_period=30):
    """
    Calculate all indicators for the entire dataset with configurable parameters
    
    Args:
        candles_5m: DataFrame with 5M candles
        candles_15m: DataFrame with 15M candles
        candles_30m: DataFrame with 30M candles
        ker_period: Period for Kaufman Efficiency Ratio (default: 10)
        kama_period: Period for KAMA initialization (default: 10)
        fast_period: Fast constant for KAMA (default: 2)
        slow_period: Slow constant for KAMA (default: 30)
        
    Returns:
        DataFrame with all calculated indicators
    """
    print(f"Processing full dataset: {len(candles_5m)} 5M candles")
    print(f"Date range: {candles_5m['timestamp'].min()} to {candles_5m['timestamp'].max()}")
    print(f"Parameters: KER Period={ker_period}, KAMA Period={kama_period}, Fast={fast_period}, Slow={slow_period}")
    
    # Create results dataframe starting with 5M data
    results = candles_5m.copy()
    
    # Add timeframe period information
    results['period_15m'] = (results['timestamp'].dt.minute % 15) // 5
    results['period_30m'] = (results['timestamp'].dt.minute % 30) // 5
    
    # Initialize result columns
    for tf in ['5m', '15m', '30m']:
        results[f'sma_{tf}'] = np.nan
        results[f'ker_{tf}'] = np.nan
        results[f'kama_{tf}'] = np.nan
    
    # Setup initial KAMA values
    last_kama = {
        '5m': None,
        '15m': None,
        '30m': None
    }
    
    # Window sizes
    window_sizes = {
        '5m': 12,  # 1 hour
        '15m': 20,  # 5 hours
        '30m': 30   # 15 hours
    }
    
    # Create lookup maps for higher timeframes
    print("Creating 15M and 30M lookup maps...")
    
    candles_15m_map = {}
    for _, row in candles_15m.iterrows():
        candles_15m_map[row['timestamp']] = row.to_dict()
    
    candles_30m_map = {}
    for _, row in candles_30m.iterrows():
        candles_30m_map[row['timestamp']] = row.to_dict()
    
    # Create windows for rolling calculations - make sure they're large enough 
    # for the largest requested period
    max_window_size = max(max(window_sizes.values()), ker_period, kama_period)
    
    windows = {
        '5m': deque(maxlen=max_window_size + 10),  # add buffer
        '15m': deque(maxlen=max_window_size + 10),
        '30m': deque(maxlen=max_window_size + 10)
    }
    
    # Sort by timestamp to ensure chronological processing
    results = results.sort_values('timestamp').reset_index(drop=True)
    
    # For each 5M candle, process indicators
    for idx, row in results.iterrows():
        current_time = row['timestamp']
        
        # Progress indicator every 1000 rows
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(results)}: {current_time}")
        
        # Add current 5M candle to window
        windows['5m'].append(row.to_dict())
        
        # Find corresponding 15M candle
        current_15m_time = current_time.floor('15min')
        if current_15m_time in candles_15m_map:
            # If this is a new 15M candle we haven't seen yet, add it to window
            if len(windows['15m']) == 0 or windows['15m'][-1]['timestamp'] != current_15m_time:
                windows['15m'].append(candles_15m_map[current_15m_time])
        
        # Find corresponding 30M candle
        current_30m_time = current_time.floor('30min')
        if current_30m_time in candles_30m_map:
            # If this is a new 30M candle we haven't seen yet, add it to window
            if len(windows['30m']) == 0 or windows['30m'][-1]['timestamp'] != current_30m_time:
                windows['30m'].append(candles_30m_map[current_30m_time])
        
        # Calculate indicators for each timeframe
        for tf, window in windows.items():
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
    
    print("Indicator calculation complete!")
    
    # Create combination indicators
    print("Creating combined indicators...")
    
    # Cross-timeframe ratios
    results['kama_ratio_5m_15m'] = results['kama_5m'] / results['kama_15m']
    results['kama_ratio_5m_30m'] = results['kama_5m'] / results['kama_30m']
    results['kama_ratio_15m_30m'] = results['kama_15m'] / results['kama_30m']
    
    results['ker_ratio_5m_15m'] = results['ker_5m'] / results['ker_15m']
    results['ker_ratio_5m_30m'] = results['ker_5m'] / results['ker_30m'] 
    results['ker_ratio_15m_30m'] = results['ker_15m'] / results['ker_30m']
    
    # Weighted averages
    results['kama_wavg'] = (3*results['kama_5m'] + 2*results['kama_15m'] + results['kama_30m'])/6
    results['ker_wavg'] = (3*results['ker_5m'] + 2*results['ker_15m'] + results['ker_30m'])/6
    
    # Trend strength indicators
    results['trend_strength'] = results['ker_5m'] * results['ker_15m'] * results['ker_30m']
    
    # Volatility-adjusted KAMA
    results['kama_vol_adj'] = results['kama_5m'] * np.sqrt(results['ker_5m'])
    
    # Rate of change (3-period)
    for tf in ['5m', '15m', '30m']:
        results[f'kama_{tf}_roc'] = results[f'kama_{tf}'].pct_change(3) * 100
        
    # Create the advanced indicator combinations from the recommendations
    
    # 1. Trend-Strength-Adjusted KAMA
    results['TSKA'] = results['kama_15m'] * (1 + results['ker_5m'] - results['ker_30m'])
    
    # 2. Adaptive Fractal Efficiency
    results['AFE'] = results['ker_5m'] * (results['kama_5m'] / results['kama_15m']) * np.sign(results['kama_15m'] - results['kama_30m'])
    
    # 3. Multi-Scale Momentum
    results['MSM'] = (
        (results['kama_5m_roc'] * results['ker_5m'] + 
         results['kama_15m_roc'] * results['ker_15m'] + 
         results['kama_30m_roc'] * results['ker_30m']) / 
        (results['ker_5m'] + results['ker_15m'] + results['ker_30m'])
    )
    
    # 4. Hierarchical Momentum
    results['hier_momentum'] = (
        np.sign(results['kama_5m_roc']) * 
        np.sign(results['kama_15m_roc']) * 
        np.sign(results['kama_30m_roc']) * 
        results[['kama_5m_roc', 'kama_15m_roc', 'kama_30m_roc']].abs().max(axis=1)
    )
    
    # 5. Adaptive Timeframe Allocation
    results['ATA'] = (
        (results['kama_5m'] * results['ker_5m'] + 
         results['kama_15m'] * results['ker_15m'] + 
         results['kama_30m'] * results['ker_30m']) /
        (results['ker_5m'] + results['ker_15m'] + results['ker_30m'])
    )
    
    return results

def analyze_indicators_simple(results, output_directory):
    """
    Perform simplified analysis of calculated indicators without advanced stats libraries
    
    Args:
        results: DataFrame with calculated indicators
        output_directory: Directory to save output files
        
    Returns:
        Correlation matrix
    """
    print("Performing simplified indicator analysis...")
    
    # List of all indicators to analyze
    indicators = [
        'sma_5m', 'sma_15m', 'sma_30m',
        'ker_5m', 'ker_15m', 'ker_30m',
        'kama_5m', 'kama_15m', 'kama_30m',
        'kama_ratio_5m_15m', 'kama_ratio_5m_30m', 'kama_ratio_15m_30m',
        'ker_ratio_5m_15m', 'ker_ratio_5m_30m', 'ker_ratio_15m_30m',
        'kama_wavg', 'ker_wavg', 'trend_strength', 'kama_vol_adj',
        'kama_5m_roc', 'kama_15m_roc', 'kama_30m_roc',
        'TSKA', 'AFE', 'MSM', 'hier_momentum', 'ATA'
    ]
    
    # Ensure indicators are valid columns
    indicators = [i for i in indicators if i in results.columns]
    
    # Calculate correlation matrix
    correlation_df = results[indicators].corr()
    
    # Save correlation matrix to CSV
    correlation_csv = os.path.join(output_directory, 'Indicator_Correlations.csv')
    correlation_df.to_csv(correlation_csv)
    print(f"Correlation matrix saved to: {correlation_csv}")
    
    # Create basic correlation heatmap without seaborn
    plt.figure(figsize=(16, 14))
    plt.imshow(correlation_df, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(indicators)), indicators, rotation=90)
    plt.yticks(range(len(indicators)), indicators)
    plt.title('Indicator Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    # Save correlation heatmap
    heatmap_file = os.path.join(output_directory, 'Correlation_Heatmap.png')
    plt.savefig(heatmap_file, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_file}")
    
    # Simple stationarity approximation based on rolling mean/std
    print("Calculating basic stationarity metrics...")
    stationarity_results = []
    
    for indicator in indicators:
        if indicator in results.columns:
            series = results[indicator].dropna()
            if len(series) < 100:
                continue
                
            # Calculate rolling means and standard deviations
            rolling_mean = series.rolling(window=50).mean()
            rolling_std = series.rolling(window=50).std()
            
            # Calculate rate of change for these statistics
            mean_change = rolling_mean.pct_change(20).abs().mean()
            std_change = rolling_std.pct_change(20).abs().mean()
            
            # Approximate stationarity (lower values indicate more stationary series)
            stationarity_metric = mean_change + std_change
            
            result = {
                'indicator': indicator,
                'mean_stability': mean_change,
                'std_stability': std_change,
                'stationarity_metric': stationarity_metric,
                'relatively_stationary': stationarity_metric < 0.05  # arbitrary threshold
            }
            stationarity_results.append(result)
    
    # Convert stationarity results to DataFrame
    stationarity_df = pd.DataFrame(stationarity_results)
    
    # Save stationarity results
    stationarity_csv = os.path.join(output_directory, 'Basic_Stationarity_Metrics.csv')
    stationarity_df.to_csv(stationarity_csv, index=False)
    print(f"Basic stationarity metrics saved to: {stationarity_csv}")
    
    # Plot the most and least stationary indicators
    if len(stationarity_df) > 0:
        stationarity_df = stationarity_df.sort_values('stationarity_metric')
        
        # Plot top 5 most stationary indicators
        most_stationary = stationarity_df.head(5)['indicator'].tolist()
        
        plt.figure(figsize=(14, 6))
        for indicator in most_stationary:
            plt.plot(results['timestamp'].iloc[::100], results[indicator].iloc[::100], label=indicator)
        plt.title('Most Stationary Indicators (sampled)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        stationary_plot = os.path.join(output_directory, 'Most_Stationary_Indicators.png')
        plt.savefig(stationary_plot, dpi=300)
        plt.close()
        
        # Plot top 5 least stationary indicators
        least_stationary = stationarity_df.tail(5)['indicator'].tolist()
        
        plt.figure(figsize=(14, 6))
        for indicator in least_stationary:
            plt.plot(results['timestamp'].iloc[::100], results[indicator].iloc[::100], label=indicator)
        plt.title('Least Stationary Indicators (sampled)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        nonstationary_plot = os.path.join(output_directory, 'Least_Stationary_Indicators.png')
        plt.savefig(nonstationary_plot, dpi=300)
        plt.close()
    
    return correlation_df

def manual_correlation_analysis(results, output_directory):
    """Perform manual analysis of highest correlations"""
    print("Finding highest correlations...")
    
    # List of all indicators
    indicators = [col for col in results.columns if col not in ['timestamp', 'period_15m', 'period_30m', 'volume']]
    
    # Calculate all pairwise correlations
    high_correlations = []
    for i, indicator1 in enumerate(indicators):
        for j, indicator2 in enumerate(indicators):
            if i < j:  # avoid duplicates and self-correlations
                if indicator1 in results.columns and indicator2 in results.columns:
                    corr = results[indicator1].corr(results[indicator2])
                    high_correlations.append((indicator1, indicator2, abs(corr)))
    
    # Sort by absolute correlation
    high_correlations.sort(key=lambda x: x[2], reverse=True)
    
    # Save top correlations to CSV
    top_correlations = pd.DataFrame(high_correlations, columns=['Indicator1', 'Indicator2', 'AbsCorrelation'])
    top_correlations.to_csv(os.path.join(output_directory, 'Top_Correlations.csv'), index=False)
    
    # Print top 20 highest correlations
    print("\nTop 20 highest correlations:")
    for i, (ind1, ind2, corr) in enumerate(high_correlations[:20]):
        print(f"{i+1}. {ind1} and {ind2}: {corr:.4f}")
    
    # Return top 5 for further analysis
    return high_correlations[:5]

def main():
    # Set up directories
    directory = r'D:\\'  # Update this path
    output_directory = directory  # Use same directory for output
    
    try:
        # Define date range for analysis
        # For full dataset, leave these as None
        # For testing, use a smaller sample
        test_mode = True
        if test_mode:
            # Use a smaller sample for testing
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # analyze 30 days
            print(f"TEST MODE ENABLED: Analyzing data from {start_date} to {end_date}")
        else:
            start_date = None
            end_date = None
        
        # Load data
        print("Loading candle data...")
        candles_5m = load_candle_data(directory, '5M', start_date, end_date)
        candles_15m = load_candle_data(directory, '15M', start_date, end_date)
        candles_30m = load_candle_data(directory, '30M', start_date, end_date)
        
        print(f"Loaded {len(candles_5m)} 5M candles, {len(candles_15m)} 15M candles, {len(candles_30m)} 30M candles")
        
        # Define calculation parameters - these can be easily adjusted
        calc_params = {
            # Default parameters
            'default': {
                'ker_period': 10, 
                'kama_period': 10, 
                'fast_period': 2, 
                'slow_period': 30
            },
            # Alternative parameter sets for experimentation
            'short_ker': {
                'ker_period': 5,
                'kama_period': 10,
                'fast_period': 2,
                'slow_period': 30
            },
            'long_kama': {
                'ker_period': 10,
                'kama_period': 20,
                'fast_period': 2,
                'slow_period': 30
            },
            'custom_fast_slow': {
                'ker_period': 10,
                'kama_period': 10,
                'fast_period': 3,
                'slow_period': 25
            }
        }
        
        # Choose which parameter set to use
        param_set = 'default'
        params = calc_params[param_set]
        
        # Calculate indicators with selected parameters
        results = calculate_indicators_for_dataset(
            candles_5m, candles_15m, candles_30m,
            ker_period=params['ker_period'],
            kama_period=params['kama_period'],
            fast_period=params['fast_period'],
            slow_period=params['slow_period']
        )
        
        # Add parameter info to filename
        param_info = f"_KER{params['ker_period']}_KAMA{params['kama_period']}_F{params['fast_period']}_S{params['slow_period']}"
        
        # Save full results
        results_file = os.path.join(output_directory, f'Full_Indicator_Results{param_info}.csv')
        results.to_csv(results_file, index=False)
        print(f"Full indicator results saved to: {results_file}")
        
        # Save a more manageable sample for quick review
        sample_size = min(10000, len(results))
        sample = results.sample(sample_size, random_state=42)
        sample_file = os.path.join(output_directory, f'Sample_Indicator_Results{param_info}.csv')
        sample.to_csv(sample_file, index=False)
        print(f"Sample results ({sample_size} rows) saved to: {sample_file}")
        
        # Perform simplified analysis
        correlation_df = analyze_indicators_simple(results, output_directory)
        
        # Manual correlation analysis
        top_correlations = manual_correlation_analysis(results, output_directory)
        
        # Summary statistics
        print("\nSummary Statistics for Key Indicators:")
        key_indicators = ['ker_5m', 'ker_15m', 'ker_30m', 'kama_5m', 'kama_15m', 'kama_30m', 
                          'TSKA', 'AFE', 'MSM', 'hier_momentum', 'ATA']
        key_indicators = [k for k in key_indicators if k in results.columns]
        print(results[key_indicators].describe())
        
        # Print recommendations for Bayesian optimization
        print("\n========== RECOMMENDATIONS FOR BAYESIAN OPTIMIZATION ==========")
        print("Based on correlation analysis:")
        print("1. Consider using indicators with low cross-correlation to avoid redundancy")
        print("2. Focus on both individual timeframe indicators (KER, KAMA) and composite indicators (TSKA, AFE, MSM)")
        print("3. Use KER values as volatility filters and weighting factors")
        print("4. The advanced composite indicators may provide better information density with fewer parameters")
        print(f"5. Current parameters: KER Period={params['ker_period']}, KAMA Period={params['kama_period']}, Fast={params['fast_period']}, Slow={params['slow_period']}")
        
        # Save parameter info to a separate file for reference
        param_file = os.path.join(output_directory, 'Parameter_Sets.txt')
        with open(param_file, 'w') as f:
            f.write("Parameter Sets for Indicator Calculations:\n\n")
            for set_name, set_params in calc_params.items():
                f.write(f"{set_name}:\n")
                for param_name, param_value in set_params.items():
                    f.write(f"  {param_name} = {param_value}\n")
                f.write("\n")
        print(f"Parameter sets saved to: {param_file}")
        
        print("\nAnalysis complete! All results saved to output directory.")
        
    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
