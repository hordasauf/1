import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

def load_and_process_fx_data():
    """Load and process GBPJPY data from CSV files"""
    # Get list of all relevant files
    file_pattern = r'D:\GBPJPY_202[45]_[0-9][0-9].csv'
    files = glob.glob(file_pattern)

    if not files:
        raise ValueError("No GBPJPY files found in D:\\ matching the pattern GBPJPY_2024_XX or GBPJPY_2025_XX")

    print(f"Found {len(files)} files to process:")
    for f in sorted(files):
        print(f"  {os.path.basename(f)}")

    # Initialize empty list to store all data
    all_data = []

    for file in sorted(files):
        print(f"\nProcessing {os.path.basename(file)}")
        try:
            # Read CSV file
            df = pd.read_csv(file, header=None,
                             names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

            # Combine date and time into timestamp
            df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                             format='%Y.%m.%d %H:%M')

            print(f"  Loaded {len(df)} rows")
            print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            all_data.append(df)

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            raise

    # Combine all data and sort by timestamp
    print("\nCombining all data...")
    full_data = pd.concat(all_data, ignore_index=True)
    full_data = full_data.sort_values('timestamp')
    print(f"Total combined rows: {len(full_data)}")

    # Create resampled data for different timeframes
    timeframes = {
        '5M': '5min',
        '15M': '15min',
        '30M': '30min'
    }

    processed_data = {}

    print("\nCreating timeframe candles...")
    for tf_name, tf_offset in timeframes.items():
        print(f"\nProcessing {tf_name} timeframe")
        # Resample to desired timeframe
        resampled = full_data.set_index('timestamp').resample(tf_offset).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        # Reset index to make timestamp a column
        resampled = resampled.reset_index()

        # Store in processed data dictionary
        processed_data[tf_name] = resampled

        print(f"  Created {len(resampled)} candles")
        print(f"  Time range: {resampled['timestamp'].min()} to {resampled['timestamp'].max()}")

    return processed_data

def cross_validate_15m_vs_5m(data_15m, data_5m):
    """Cross validate 15M candles against their constituent 5M candles"""
    print("\nCross-validating 15M candles against 5M candles...")

    mismatches = []
    total_checked = 0

    # For each 15M candle
    for idx, candle_15m in data_15m.iterrows():
        total_checked += 1
        start_time = candle_15m['timestamp']
        end_time = start_time + pd.Timedelta(minutes=15)

        # Get constituent 5M candles
        mask_5m = (data_5m['timestamp'] >= start_time) & (data_5m['timestamp'] < end_time)
        constituent_5m = data_5m[mask_5m]

        if len(constituent_5m) == 0:
            continue

        # Calculate expected OHLC from 5M candles
        expected_15m = {
            'Open': constituent_5m.iloc[0]['Open'],
            'High': constituent_5m['High'].max(),
            'Low': constituent_5m['Low'].min(),
            'Close': constituent_5m.iloc[-1]['Close'],
            'Volume': constituent_5m['Volume'].sum()
        }

        # Check for mismatches
        tolerance = 0.00001

        if (abs(expected_15m['Open'] - candle_15m['Open']) > tolerance or
                abs(expected_15m['High'] - candle_15m['High']) > tolerance or
                abs(expected_15m['Low'] - candle_15m['Low']) > tolerance or
                abs(expected_15m['Close'] - candle_15m['Close']) > tolerance or
                abs(expected_15m['Volume'] - candle_15m['Volume']) > tolerance):

            mismatches.append({
                'timestamp': start_time,
                'expected': expected_15m,
                'actual': candle_15m,
                'constituent_5m': constituent_5m
            })

    # Report results
    print(f"\nChecked {total_checked} 15M candles")
    if not mismatches:
        print("✓ All 15M candles match their constituent 5M candles!")
    else:
        print(f"\n❌ Found {len(mismatches)} mismatches:")
        for mismatch in mismatches[:5]:  # Show first 5 mismatches
            print(f"\nMismatch at {mismatch['timestamp']}:")
            print("\nExpected (calculated from 5M):")
            print(f"O: {mismatch['expected']['Open']:.5f}")
            print(f"H: {mismatch['expected']['High']:.5f}")
            print(f"L: {mismatch['expected']['Low']:.5f}")
            print(f"C: {mismatch['expected']['Close']:.5f}")
            print(f"V: {mismatch['expected']['Volume']}")

            print("\nActual 15M candle:")
            print(f"O: {mismatch['actual']['Open']:.5f}")
            print(f"H: {mismatch['actual']['High']:.5f}")
            print(f"L: {mismatch['actual']['Low']:.5f}")
            print(f"C: {mismatch['actual']['Close']:.5f}")
            print(f"V: {mismatch['actual']['Volume']}")

            print("\nConstituent 5M candles:")
            print(mismatch['constituent_5m'][['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']])
            print("\n" + "="*50)

    return mismatches

def analyze_time_gaps(df, timeframe):
    """Analyze gaps in time series data"""
    minutes = int(timeframe.replace('M', ''))
    expected_diff = pd.Timedelta(minutes=minutes)
    time_diffs = df['timestamp'].diff()

    # Find gaps
    gaps = df[time_diffs != expected_diff].copy()
    if len(gaps) == 0:
        print(f"\n{timeframe}: No time gaps found")
        return

    # Analyze gaps
    gaps['prev_timestamp'] = gaps['timestamp'].shift(1)
    gaps['gap_duration'] = gaps['timestamp'] - gaps['prev_timestamp']

    # Group gaps by duration
    gap_groups = gaps.groupby('gap_duration').size()

    print(f"\n{timeframe} Time Gap Analysis:")
    print(f"Total gaps found: {len(gaps)}")
    print("\nGap durations:")
    for duration, count in gap_groups.items():
        if pd.isna(duration):
            continue
        days = duration.days
        hours = duration.components.hours
        minutes = duration.components.minutes
        print(f"  {days}d {hours}h {minutes}m: {count} occurrences")

def validate_edge_cases(processed_data):
    """Validate candle alignment at week boundaries and other edge cases"""
    print("\nValidating edge case alignments...")

    data_5m = processed_data['5M']
    data_15m = processed_data['15M']
    data_30m = processed_data['30M']

    # Find all week endings
    week_endings = data_5m[data_5m['timestamp'].dt.dayofweek == 4]  # Friday
    week_endings = week_endings[week_endings['timestamp'].dt.hour == 23]  # Last hour

    for _, last_5m_candle in week_endings.iterrows():
        last_time = last_5m_candle['timestamp']
        print(f"\nChecking alignment for week ending: {last_time}")

        # Get the 15M period that should contain this 5M candle
        period_15m_start = last_time.replace(minute=(last_time.minute // 15) * 15)
        containing_15m = data_15m[data_15m['timestamp'] == period_15m_start]

        # Get the 30M period that should contain this 5M candle
        period_30m_start = last_time.replace(minute=(last_time.minute // 30) * 30)
        containing_30m = data_30m[data_30m['timestamp'] == period_30m_start]

        # Get all 5M candles that should be in the last 15M candle
        mask_15m = (data_5m['timestamp'] >= period_15m_start) & (data_5m['timestamp'] <= last_time)
        constituent_5m_for_15m = data_5m[mask_15m]

        # Get all 5M candles that should be in the last 30M candle
        mask_30m = (data_5m['timestamp'] >= period_30m_start) & (data_5m['timestamp'] <= last_time)
        constituent_5m_for_30m = data_5m[mask_30m]

        print(f"\n5M candle at: {last_time}")
        print(f"OHLC: {last_5m_candle['Open']:.3f}, {last_5m_candle['High']:.3f}, "
              f"{last_5m_candle['Low']:.3f}, {last_5m_candle['Close']:.3f}")

        if not containing_15m.empty:
            print(f"\nContaining 15M candle ({period_15m_start}):")
            print(f"Number of constituent 5M candles: {len(constituent_5m_for_15m)}")
            print(f"OHLC: {containing_15m.iloc[0]['Open']:.3f}, {containing_15m.iloc[0]['High']:.3f}, "
                  f"{containing_15m.iloc[0]['Low']:.3f}, {containing_15m.iloc[0]['Close']:.3f}")

            # Verify 15M candle matches its constituents
            calc_15m = {
                'Open': constituent_5m_for_15m.iloc[0]['Open'],
                'High': constituent_5m_for_15m['High'].max(),
                'Low': constituent_5m_for_15m['Low'].min(),
                'Close': constituent_5m_for_15m.iloc[-1]['Close']
            }
            print("\n15M candle verification:")
            print(f"Expected from 5M: O:{calc_15m['Open']:.3f}, H:{calc_15m['High']:.3f}, "
                  f"L:{calc_15m['Low']:.3f}, C:{calc_15m['Close']:.3f}")
        else:
            print("Warning: No containing 15M candle found!")

        if not containing_30m.empty:
            print(f"\nContaining 30M candle ({period_30m_start}):")
            print(f"Number of constituent 5M candles: {len(constituent_5m_for_30m)}")
            print(f"OHLC: {containing_30m.iloc[0]['Open']:.3f}, {containing_30m.iloc[0]['High']:.3f}, "
                  f"{containing_30m.iloc[0]['Low']:.3f}, {containing_30m.iloc[0]['Close']:.3f}")

            # Verify 30M candle matches its constituents
            calc_30m = {
                'Open': constituent_5m_for_30m.iloc[0]['Open'],
                'High': constituent_5m_for_30m['High'].max(),
                'Low': constituent_5m_for_30m['Low'].min(),
                'Close': constituent_5m_for_30m.iloc[-1]['Close']
            }
            print("\n30M candle verification:")
            print(f"Expected from 5M: O:{calc_30m['Open']:.3f}, H:{calc_30m['High']:.3f}, "
                  f"L:{calc_30m['Low']:.3f}, C:{calc_30m['Close']:.3f}")
        else:
            print("Warning: No containing 30M candle found!")

        print("\n" + "="*80)

def basic_sanity_checks(processed_data):
    """Run basic sanity checks on the data"""
    print("\nRunning basic sanity checks...")
    for tf in ['5M', '15M', '30M']:
        df = processed_data[tf]
        print(f"\n{tf} Checks:")
        print(f"Total candles: {len(df)}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Any High < Low: {(df['High'] < df['Low']).any()}")
        print(f"Any Close outside High-Low range: {((df['Close'] > df['High']) | (df['Close'] < df['Low'])).any()}")
        print(f"Any Open outside High-Low range: {((df['Open'] > df['High']) | (df['Open'] < df['Low'])).any()}")

if __name__ == "__main__":
    try:
        print("Starting GBPJPY data processing...")
        processed_data = load_and_process_fx_data()

        # Run all validations
        basic_sanity_checks(processed_data)

        # Cross validate 15M vs 5M
        cross_validate_15m_vs_5m(processed_data['15M'], processed_data['5M'])

        # Analyze gaps for each timeframe
        for tf in ['5M', '15M', '30M']:
            analyze_time_gaps(processed_data[tf], tf)

        # Check edge cases
        validate_edge_cases(processed_data)

        print("\nProcessing completed successfully!")

        save_choice = input("\nWould you like to save the processed data? (y/n): ")
        if save_choice.lower() == 'y':
            save_path = r'D:\processed_gbpjpy_data.pkl'
            pd.to_pickle(processed_data, save_path)
            print(f"Data saved to {save_path}")
            print("\nTo load this data later, use:")
            print("import pandas as pd")
            print("processed_data = pd.read_pickle(r'D:\\processed_gbpjpy_data.pkl')")
            print("\nThe data will be loaded into a dictionary with keys '5M', '15M', and '30M'")
            print("Each timeframe contains a DataFrame with columns: timestamp, Open, High, Low, Close, Volume")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        input("\nPress Enter to exit...")
