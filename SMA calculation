import pandas as pd
import os
from collections import deque
from datetime import datetime, timedelta

def load_candle_data(directory, timeframe):
    """Load and validate candle data with numeric conversion"""
    file_path = os.path.join(directory, f'GBPJPY_{timeframe}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    df = pd.read_csv(file_path)

    # Convert and validate data types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Data quality checks
    if df[numeric_cols].isnull().values.any():
        raise ValueError("Non-numeric values found in price/volume data")

    return df.sort_values('timestamp').reset_index(drop=True)

def get_third_weekday_of_month(year, month):
    """Get the third weekday of a given month and year"""
    day = pd.Timestamp(year=year, month=month, day=1)
    weekday_count = 0
    while True:
        if day.weekday() < 5:  # Monday-Friday
            weekday_count += 1
            if weekday_count == 3:
                return day.replace(hour=10, minute=0, second=0, microsecond=0)
        day += timedelta(days=1)

def calculate_inprogress_candle(base_data, current_time, timeframe):
    """
    Calculate in-progress candle based on current 5M data

    Args:
        base_data: DataFrame with 5M candles
        current_time: Current timestamp
        timeframe: Timeframe string (e.g., '15M', '30M')

    Returns:
        Dictionary with candle OHLC data or None if no data
    """
    timeframe_mins = int(timeframe.replace('M', ''))

    # Calculate the start of the current period
    period_start = current_time.floor(f'{timeframe_mins}T')

    # Get all 5M candles that contribute to this period
    # Important: we use < current_time (exclusive) to ensure we only include
    # completed 5M candles that would be visible at the exact moment
    mask = (base_data['timestamp'] >= period_start) & (base_data['timestamp'] <= current_time)
    period_data = base_data[mask]

    if not period_data.empty:
        return {
            'timestamp': period_start,
            'open': float(period_data.iloc[0]['open']),
            'high': float(period_data['high'].max()),
            'low': float(period_data['low'].min()),
            'close': float(period_data.iloc[-1]['close']),
            'volume': float(period_data['volume'].sum())
        }
    return None

def calculate_sma(candles, window_size):
    """
    Calculate Simple Moving Average from a list of candles

    Args:
        candles: List of candle dictionaries
        window_size: Number of periods for SMA calculation

    Returns:
        SMA value or None if insufficient data
    """
    if not candles or len(candles) < window_size:
        return None

    try:
        closes = [float(c['close']) for c in candles[-window_size:]]
        return sum(closes) / window_size
    except (TypeError, ValueError, KeyError) as e:
        print(f"Error calculating SMA: {e}")
        print(f"Candles data: {candles[-window_size:]}")
        return None

def get_prior_data(df, start_date, periods):
    """Get prior data for verification"""
    return df[df['timestamp'] < start_date].tail(periods)

def process_time_window(candles_5m, candles_15m, candles_30m, start_date, end_date):
    """
    Process time window with precise handling of in-progress vs completed candles

    Args:
        candles_5m: DataFrame with 5M candles
        candles_15m: DataFrame with 15M candles
        candles_30m: DataFrame with 30M candles
        start_date: Start timestamp for analysis (e.g., 10:00)
        end_date: End timestamp for analysis (e.g., 11:00)

    Returns:
        DataFrame with results
    """
    # Filter 5M candles for the analysis period
    analysis_5m = candles_5m[(candles_5m['timestamp'] >= start_date) &
                             (candles_5m['timestamp'] < end_date)]

    results = []

    # Cache for SMA calculations by timeframe period
    sma_cache = {
        '15M': {},  # Will store SMA values by 15M period (0, 1, 2)
        '30M': {}   # Will store SMA values by 30M period (0-5)
    }

    for idx, row in analysis_5m.iterrows():
        current_time = row['timestamp']
        current_price = float(row['close'])

        # Calculate which period within 15M and 30M timeframes
        period_15m = (current_time.minute % 15) // 5
        period_30m = (current_time.minute % 30) // 5
        current_hour = current_time.hour

        # Calculate 5M SMA (1-hour window, 12 periods) - this always updates
        window_5m = candles_5m[candles_5m['timestamp'] < current_time].tail(12).to_dict('records')
        sma_5m = calculate_sma(window_5m, min(len(window_5m), 12))
        if sma_5m is not None:
            sma_5m = round(sma_5m, 5)

        # Create a unique key for this 15M and 30M period in this hour
        key_15m = f"{current_hour}:{period_15m}"
        key_30m = f"{current_hour}:{period_30m}"

        # For 15M SMA - check cache first, calculate only if needed
        if key_15m in sma_cache['15M']:
            sma_15m = sma_cache['15M'][key_15m]
        else:
            # Calculate the appropriate window based on period
            if period_15m == 0:
                # For XX:00, XX:15, XX:30, XX:45 - use 19 completed + new in-progress
                completed_15m = candles_15m[candles_15m['timestamp'] < current_time.floor('15min')].tail(19).to_dict('records')
                inprogress_15m = {
                    'timestamp': current_time.floor('15min'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                window_15m = completed_15m + [inprogress_15m]

            elif period_15m == 1:
                # For XX:05, XX:20, XX:35, XX:50 - use 19 completed + in-progress with combined 5M data
                completed_15m = candles_15m[candles_15m['timestamp'] < current_time.floor('15min')].tail(19).to_dict('records')

                # Get all 5M data from start of this 15M period to current time
                period_start = current_time.floor('15min')
                previous_5m = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                         (candles_5m['timestamp'] <= current_time)]

                if not previous_5m.empty:
                    inprogress_15m = {
                        'timestamp': period_start,
                        'open': float(previous_5m.iloc[0]['open']),
                        'high': float(previous_5m['high'].max()),
                        'low': float(previous_5m['low'].min()),
                        'close': float(previous_5m.iloc[-1]['close']),
                        'volume': float(previous_5m['volume'].sum())
                    }
                    window_15m = completed_15m + [inprogress_15m]
                else:
                    window_15m = completed_15m

            elif period_15m == 2:
                # For XX:10, XX:25, XX:40, XX:55 - special case, use fully completed 15M candle
                # Calculate the completed XX:00/XX:15/XX:30/XX:45 15M candle using all 5M data
                period_start = current_time.floor('15min')
                period_end = period_start + timedelta(minutes=15)

                # Create a more precise window for accessing 5M data
                # We need all data from period_start up to but not including period_end
                # But this might include data after current_time (which we don't want)
                # So we use min(period_end, current_time + timedelta(seconds=1))
                effective_end = min(period_end, current_time + timedelta(seconds=1))

                mask = (candles_5m['timestamp'] >= period_start) & (candles_5m['timestamp'] < effective_end)
                period_data = candles_5m[mask]

                if not period_data.empty:
                    completed_15m_candle = {
                        'timestamp': period_start,
                        'open': float(period_data.iloc[0]['open']),
                        'high': float(period_data['high'].max()),
                        'low': float(period_data['low'].min()),
                        'close': float(period_data.iloc[-1]['close']),
                        'volume': float(period_data['volume'].sum())
                    }

                    # Use prior 19 completed candles + the freshly completed one
                    prior_candles = candles_15m[candles_15m['timestamp'] < period_start].tail(19).to_dict('records')
                    window_15m = prior_candles + [completed_15m_candle]
                else:
                    # Fallback if data is missing
                    window_15m = candles_15m[candles_15m['timestamp'] < current_time].tail(20).to_dict('records')

            else:
                # Fallback case (shouldn't happen)
                window_15m = candles_15m[candles_15m['timestamp'] < current_time].tail(20).to_dict('records')

            # Calculate and cache the 15M SMA
            sma_15m = calculate_sma(window_15m, min(len(window_15m), 20))
            if sma_15m is not None:
                sma_15m = round(sma_15m, 5)
                sma_cache['15M'][key_15m] = sma_15m

        # For 30M SMA - check cache first, calculate only if needed
        if key_30m in sma_cache['30M']:
            sma_30m = sma_cache['30M'][key_30m]
        else:
            # Calculate the appropriate window based on period
            if period_30m == 0:
                # For XX:00, XX:30 - use 29 completed + new in-progress
                completed_30m = candles_30m[candles_30m['timestamp'] < current_time.floor('30min')].tail(29).to_dict('records')
                inprogress_30m = {
                    'timestamp': current_time.floor('30min'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                window_30m = completed_30m + [inprogress_30m]

            elif period_30m == 5:
                # For XX:25, XX:55 - special case, use fully completed 30M candle
                period_start = current_time.floor('30min')
                period_end = period_start + timedelta(minutes=30)

                # Create an effective end time that doesn't go beyond current time
                effective_end = min(period_end, current_time + timedelta(seconds=1))

                mask = (candles_5m['timestamp'] >= period_start) & (candles_5m['timestamp'] < effective_end)
                period_data = candles_5m[mask]

                if not period_data.empty:
                    completed_30m_candle = {
                        'timestamp': period_start,
                        'open': float(period_data.iloc[0]['open']),
                        'high': float(period_data['high'].max()),
                        'low': float(period_data['low'].min()),
                        'close': float(period_data.iloc[-1]['close']),
                        'volume': float(period_data['volume'].sum())
                    }

                    # Use prior 29 completed candles + the freshly completed one
                    prior_candles = candles_30m[candles_30m['timestamp'] < period_start].tail(29).to_dict('records')
                    window_30m = prior_candles + [completed_30m_candle]
                else:
                    # Fallback if data is missing
                    window_30m = candles_30m[candles_30m['timestamp'] < current_time].tail(30).to_dict('records')

            else:
                # For all other periods - use in-progress candle with combined 5M data
                completed_30m = candles_30m[candles_30m['timestamp'] < current_time.floor('30min')].tail(29).to_dict('records')

                # Get all 5M data from start of this 30M period to current time
                period_start = current_time.floor('30min')
                previous_5m = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                         (candles_5m['timestamp'] <= current_time)]

                if not previous_5m.empty:
                    inprogress_30m = {
                        'timestamp': period_start,
                        'open': float(previous_5m.iloc[0]['open']),
                        'high': float(previous_5m['high'].max()),
                        'low': float(previous_5m['low'].min()),
                        'close': float(previous_5m.iloc[-1]['close']),
                        'volume': float(previous_5m['volume'].sum())
                    }
                    window_30m = completed_30m + [inprogress_30m]
                else:
                    window_30m = completed_30m

            # Calculate and cache the 30M SMA
            sma_30m = calculate_sma(window_30m, min(len(window_30m), 30))
            if sma_30m is not None:
                sma_30m = round(sma_30m, 5)
                sma_cache['30M'][key_30m] = sma_30m

        # Save results
        results.append({
            'timestamp': current_time,
            'price': current_price,
            'sma_5m': sma_5m,
            'sma_15m': sma_15m,
            'sma_30m': sma_30m,
            '15M_period': period_15m,
            '30M_period': period_30m
        })

    return pd.DataFrame(results)

    for idx, row in analysis_5m.iterrows():
        current_time = row['timestamp']
        current_price = float(row['close'])

        # Calculate 5M SMA (1-hour window, 12 periods)
        window_5m = candles_5m[candles_5m['timestamp'] < current_time].tail(12).to_dict('records')

        # Calculate precise SMA for each timeframe based on what would be visible
        # on a trader's screen at this exact moment

        # For 15M SMA (5-hour window, 20 periods)
        # Get current minute within the 15M period
        current_15m_minute = current_time.minute % 15

        # Calculate which 15M candle is completing at the same time as this 5M candle
        # This is crucial for the XX:10 timestamps which complete at XX:14:59,
        # the same time as the 15M candle that starts at XX:00
        current_15m_period_start = current_time.floor('15min')
        next_15m_period_start = current_15m_period_start + timedelta(minutes=15)

        if current_15m_minute == 10:
            # For XX:10 timestamps (which complete at XX:14:59)
            # Use the fully completed 15M candle that also completes at XX:14:59
            # This is the XX:00 candle, which we need to calculate using all 5M data in that period

            # Calculate the completed XX:00 15M candle using all the 5M data
            period_start = current_15m_period_start
            period_end = next_15m_period_start
            mask = (candles_5m['timestamp'] >= period_start) & (candles_5m['timestamp'] < period_end)
            period_data = candles_5m[mask]

            if not period_data.empty:
                completed_15m_candle = {
                    'timestamp': period_start,
                    'open': float(period_data.iloc[0]['open']),
                    'high': float(period_data['high'].max()),
                    'low': float(period_data['low'].min()),
                    'close': float(period_data.iloc[-1]['close']),
                    'volume': float(period_data['volume'].sum())
                }

                # Use prior 19 completed candles + the freshly completed one that finishes at XX:14:59
                prior_candles = candles_15m[candles_15m['timestamp'] < period_start].tail(19).to_dict('records')
                window_15m = prior_candles + [completed_15m_candle]
            else:
                # Fallback if data is missing
                window_15m = candles_15m[candles_15m['timestamp'] < current_time].tail(20).to_dict('records')

        elif current_15m_minute == 0:
            # For XX:00 timestamps - these complete at XX:04:59
            # Use 19 completed 15M candles + in-progress 15M candle
            # At this point, the in-progress candle only contains this single 5M bar
            completed_15m = candles_15m[candles_15m['timestamp'] < current_time.floor('15min')].tail(19).to_dict('records')

            # For XX:00, the in-progress candle is just starting, so we use the current 5M data directly
            inprogress_15m = {
                'timestamp': current_time.floor('15min'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            window_15m = completed_15m + [inprogress_15m]

        elif current_15m_minute == 5:
            # For XX:05 timestamps - these complete at XX:09:59
            # For these rows, we need to integrate both the XX:00 5M data (which locks at XX:04:59)
            # and the XX:05 5M data (which locks at XX:09:59) into the in-progress 15M candle

            # Get prior 19 completed candles
            completed_15m = candles_15m[candles_15m['timestamp'] < current_time.floor('15min')].tail(19).to_dict('records')

            # Create a more accurate in-progress 15M candle by combining the 5M data
            # from the start of the 15M period up to the current time
            period_start = current_time.floor('15min')  # XX:00
            previous_5m = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                     (candles_5m['timestamp'] < current_time)]

            if not previous_5m.empty:
                inprogress_15m = {
                    'timestamp': period_start,
                    'open': float(previous_5m.iloc[0]['open']),
                    'high': float(previous_5m['high'].max()),
                    'low': float(previous_5m['low'].min()),
                    'close': float(previous_5m.iloc[-1]['close']),
                    'volume': float(previous_5m['volume'].sum())
                }
                window_15m = completed_15m + [inprogress_15m]
            else:
                # Fallback to standard calculation if data is missing
                inprogress_15m = calculate_inprogress_candle(candles_5m, current_time, '15M')
                if inprogress_15m:
                    window_15m = completed_15m + [inprogress_15m]
                else:
                    window_15m = completed_15m

        else:
            # For all other timestamps - use proper completed candles
            window_15m = candles_15m[candles_15m['timestamp'] < current_time].tail(20).to_dict('records')

        # For 30M SMA (15-hour window, 30 periods)
        # Similar logic as 15M
        current_30m_minute = current_time.minute % 30
        current_30m_period_start = current_time.floor('30min')
        next_30m_period_start = current_30m_period_start + timedelta(minutes=30)

        # Special case for xx:25 timestamps that complete at xx:29:59
        # same time as the 30M candle that starts at xx:00
        if current_30m_minute == 25:
            # Calculate the completed XX:00 30M candle using all the 5M data
            period_start = current_30m_period_start
            period_end = next_30m_period_start
            mask = (candles_5m['timestamp'] >= period_start) & (candles_5m['timestamp'] < period_end)
            period_data = candles_5m[mask]

            if not period_data.empty:
                completed_30m_candle = {
                    'timestamp': period_start,
                    'open': float(period_data.iloc[0]['open']),
                    'high': float(period_data['high'].max()),
                    'low': float(period_data['low'].min()),
                    'close': float(period_data.iloc[-1]['close']),
                    'volume': float(period_data['volume'].sum())
                }

                # Use prior 29 completed candles + the freshly completed one
                prior_candles = candles_30m[candles_30m['timestamp'] < period_start].tail(29).to_dict('records')
                window_30m = prior_candles + [completed_30m_candle]
            else:
                # Fallback if data is missing
                window_30m = candles_30m[candles_30m['timestamp'] < current_time].tail(30).to_dict('records')
        elif current_30m_minute < 25:  # First part of 30M candle
            # For these timestamps, we need to integrate all available 5M data within the current 30M window
            completed_30m = candles_30m[candles_30m['timestamp'] < current_time.floor('30min')].tail(29).to_dict('records')

            # Get all 5M data from the start of this 30M period to current time
            period_start = current_time.floor('30min')
            previous_5m = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                     (candles_5m['timestamp'] < current_time)]

            if not previous_5m.empty:
                inprogress_30m = {
                    'timestamp': period_start,
                    'open': float(previous_5m.iloc[0]['open']),
                    'high': float(previous_5m['high'].max()),
                    'low': float(previous_5m['low'].min()),
                    'close': float(previous_5m.iloc[-1]['close']),
                    'volume': float(previous_5m['volume'].sum())
                }
                window_30m = completed_30m + [inprogress_30m]
            else:
                # Fallback to standard calculation if data is missing
                inprogress_30m = calculate_inprogress_candle(candles_5m, current_time, '30M')
                if inprogress_30m:
                    window_30m = completed_30m + [inprogress_30m]
                else:
                    window_30m = completed_30m
        else:  # Second part
            window_30m = candles_30m[candles_30m['timestamp'] < current_time].tail(30).to_dict('records')

        # Calculate 5M SMA - this updates every 5M as expected
        sma_5m = calculate_sma(window_5m, min(len(window_5m), 12))
        if sma_5m is not None:
            sma_5m = round(sma_5m, 5)

        # For 15M SMA, we need to determine if it should update based on candle boundaries
        # Determine the 15M candle boundary this 5M belongs to
        current_15m_boundary = (current_time.minute % 15) // 5

        # If we've already calculated an SMA for this 15M boundary, reuse it
        if last_sma['15M']['timestamp'] is not None and current_15m_boundary == last_sma['15M']['timestamp'].minute % 15 // 5:
            sma_15m = last_sma['15M']['value']
        else:
            # Otherwise, calculate a new SMA
            sma_15m = calculate_sma(window_15m, min(len(window_15m), 20))
            if sma_15m is not None:
                sma_15m = round(sma_15m, 5)
                # Save this SMA for potential reuse
                last_sma['15M'] = {'timestamp': current_time, 'value': sma_15m}

        # For 30M SMA, similar logic
        current_30m_boundary = (current_time.minute % 30) // 5

        if last_sma['30M']['timestamp'] is not None and current_30m_boundary == last_sma['30M']['timestamp'].minute % 30 // 5:
            sma_30m = last_sma['30M']['value']
        else:
            sma_30m = calculate_sma(window_30m, min(len(window_30m), 30))
            if sma_30m is not None:
                sma_30m = round(sma_30m, 5)
                last_sma['30M'] = {'timestamp': current_time, 'value': sma_30m}

        # Save results
        results.append({
            'timestamp': current_time,
            'price': current_price,
            'sma_5m': sma_5m,
            'sma_15m': sma_15m,
            'sma_30m': sma_30m
        })

    return pd.DataFrame(results)

def print_window_debug(window, window_type):
    """Helper function to print debug info about an SMA window"""
    if not window:
        print(f"{window_type} window is empty")
        return

    print(f"\n{window_type} Window Analysis:")
    print(f"Window size: {len(window)}")

    if len(window) > 0:
        first_ts = window[0]['timestamp'] if 'timestamp' in window[0] else "N/A"
        last_ts = window[-1]['timestamp'] if 'timestamp' in window[-1] else "N/A"
        print(f"Time range: {first_ts} to {last_ts}")

        # Print last few candles for verification
        print("\nLast 5 candles in window:")
        for i, candle in enumerate(window[-5:]):
            print(f"{i+1}: {candle}")

# Main Execution
def main():
    directory = r'D:\\'  # Update this path to your data directory

    try:
        # Load data with validation
        print("Loading data...")
        candles_5m = load_candle_data(directory, '5M')
        candles_15m = load_candle_data(directory, '15M')
        candles_30m = load_candle_data(directory, '30M')

        # Set analysis date (third weekday of November 2024)
        analysis_date = get_third_weekday_of_month(2024, 11)
        end_date = analysis_date + timedelta(hours=1)
        print(f"\nAnalysis Period: {analysis_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")

        # Print prior data for verification
        print("\nPrior 1 Hour 5M Data:")
        print(get_prior_data(candles_5m, analysis_date, 12))

        print("\nPrior 5 Hours 15M Data:")
        print(get_prior_data(candles_15m, analysis_date, 20))

        print("\nPrior 15 Hours 30M Data:")
        print(get_prior_data(candles_30m, analysis_date, 30))

        # Process data with corrected SMA logic
        print("\nProcessing indicators...")
        results = process_time_window(candles_5m, candles_15m, candles_30m, analysis_date, end_date)

        # Save and display results
        output_path = os.path.join(directory, 'SMA_Results.csv')
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        print("\nFinal Results:")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.float_format', '{:.5f}'.format)
        print(results)

        # Print a simplified version to verify SMA updates properly
        print("\nSMA Update Pattern (should show distinct values for each timeframe boundary):")
        print(results[['timestamp', 'price', 'sma_5m', 'sma_15m', 'sma_30m', '15M_period', '30M_period']])

        print("\nData Validation:")
        print(f"Price Range: {results['price'].min():.3f} - {results['price'].max():.3f}")
        print("5M SMA Values:", results['sma_5m'].describe())
        print("15M SMA Values:", results['sma_15m'].describe())
        print("30M SMA Values:", results['sma_30m'].describe())

        # Check for consistent SMA values within each timeframe period
        print("\nVerifying consistency within timeframe periods:")

        # Check 15M SMA consistency within each period
        print("\n15M SMA Consistency Check:")
        for period in [0, 1, 2]:
            period_data = results[results['15M_period'] == period]
            unique_values = period_data['sma_15m'].unique()
            print(f"Period {period}: {len(unique_values)} unique values - {unique_values}")

        # Check 30M SMA consistency within each period
        print("\n30M SMA Consistency Check:")
        for period in range(6):
            period_data = results[results['30M_period'] == period]
            unique_values = period_data['sma_30m'].unique()
            print(f"Period {period}: {len(unique_values)} unique values - {unique_values}")

        # Print raw data again for easier verification
        print("\nPrior 1 Hour 5M Data:")
        print(get_prior_data(candles_5m, analysis_date, 12))

    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        print(traceback.format_exc())

    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
