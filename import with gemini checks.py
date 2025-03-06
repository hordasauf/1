```python
"""
This script loads, processes, and aggregates 1-minute Forex data from CSV files into 5-minute, 15-minute, and 30-minute candlestick data.

It includes robust error handling and data consistency checks to ensure reliable data processing.

Author: [Your Name]
Date: [Date]
"""

import pandas as pd
import os

def load_csv_files(directory):
    """Load all CSV files from the specified directory into a single DataFrame."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

    if not all_files:
        raise ValueError(f"No CSV files found in directory '{directory}'.")

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, header=None, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file '{file}': {e}")
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def parse_timestamps(df):
    """Combine date and time columns into a single datetime column."""
    if not all(col in df.columns for col in ['date', 'time']):
        raise ValueError("CSV files must contain 'date' and 'time' columns.")

    #print(df.iloc[125120:125125]) #Uncomment this line to print data around the error row for debugging.
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y.%m.%d %H:%M', errors='coerce')
    df.drop(columns=['date', 'time'], inplace=True)
    df = df.dropna(subset=['timestamp'])
    df = df.drop_duplicates(subset=['timestamp'])

    if not df['timestamp'].is_monotonic_increasing:
        raise ValueError("Timestamps are not in ascending order.")

    return df

def handle_missing_data(df):
    """Fill missing timestamps by carrying forward the previous OHLC values."""
    full_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='T')
    df = df.set_index('timestamp').reindex(full_range).ffill().reset_index()
    df.rename(columns={'index': 'timestamp'}, inplace=True)
    return df

def aggregate_candles(df, timeframe):
    """Aggregate 1-minute data into the specified timeframe."""
    timeframe_minutes = int(timeframe.replace('M', ''))
    df['period_start'] = df['timestamp'].dt.floor(f'{timeframe_minutes}T')
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    aggregated = df.groupby('period_start').agg(agg_dict).reset_index()
    aggregated.rename(columns={'period_start': 'timestamp'}, inplace=True)

    # Data Range Consistency Check
    if not (aggregated['high'] >= aggregated['low']).all():
        raise ValueError("High values are not consistently greater than or equal to low values.")
    if not ((aggregated['open'] >= aggregated['low']) & (aggregated['open'] <= aggregated['high'])).all():
        raise ValueError("Open values are not consistently within the high-low range.")
    if not ((aggregated['close'] >= aggregated['low']) & (aggregated['close'] <= aggregated['high'])).all():
        raise ValueError("Close values are not consistently within the high-low range.")

    return aggregated

def check_numeric_data(df):
    """Check if OHLCV columns are numeric."""
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric.")

# Directory containing CSV files
directory = r'D:\' # Change this to your directory path

try:
    raw_data = load_csv_files(directory)
    check_numeric_data(raw_data)
    parsed_data = parse_timestamps(raw_data)
    filled_data = handle_missing_data(parsed_data)
    candles_5m = aggregate_candles(filled_data, '5M')
    candles_15m = aggregate_candles(filled_data, '15M')
    candles_30m = aggregate_candles(filled_data, '30M')

    candles_5m.to_csv(r'D:\GBPJPY_5M.csv', index=False)
    candles_15m.to_csv(r'D:\GBPJPY_15M.csv', index=False)
    candles_30m.to_csv(r'D:\GBPJPY_30M.csv', index=False)

    print("5M Candles:")
    print(candles_5m.head())
    print("\n15M Candles:")
    print(candles_15m.head())
    print("\n30M Candles:")
    print(candles_30m.head())

except ValueError as e:
    print(f"Error: {e}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Verification Script for Generated CSVs
def verify_candle_files(directory):
    """Verify the consistency of the generated candle CSV files."""
    files = ['GBPJPY_5M.csv', 'GBPJPY_15M.csv', 'GBPJPY_30M.csv']
    for file in files:
        file_path = os.path.join(directory, file)

        if not os.path.exists(file_path):
            print(f"Error: File '{file}' does not exist.")
            continue

        try:
            df = pd.read_csv(file_path)
        except pd.errors.ParserError as e:
            print(f"Error: Parsing file '{file}' failed: {e}")
            continue

        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: File '{file}' is missing required columns.")
            continue

        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except ValueError:
            print(f"Error: File '{file}' timestamp column is not in a valid datetime format.")
            continue

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Error: File '{file}' column '{col}' is not numeric.")
                continue

        if not df['timestamp'].is_monotonic_increasing:
            print(f"Error: File '{file}' timestamps are not in ascending order.")
            continue

        if not (df['high'] >= df['low']).all():
            print(f"Error: File '{file}' high values are not consistently greater than or equal to low values.")
            continue
        if not ((df['open'] >= df['low']) & (df['open'] <= df['high'])).all():
            print(f"Error: File '{file}' open values are not consistently within the high-low range.")
            continue
        if not ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all():
            print(f"Error: File '{file}' close values are not consistently within the high-low range.")
            continue
        if df.isnull().values.any():
            print(f"Error: File '{file}' contains NaN values.")
            continue

        print(f"File '{file}' passed all checks.")

# Directory containing the generated CSV files
verify_candle_files(directory)
```
