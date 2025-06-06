import pandas as pd
import os
import csv
from datetime import datetime, timedelta

def load_candle_data(directory, timeframe):
    """Load candle data from CSV"""
    file_path = os.path.join(directory, f'GBPJPY_{timeframe}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
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

def calculate_weighted_close(high, low, close, kama_style=True):
    """Calculate weighted close price"""
    if kama_style:
        # (High + Low + Close*2) / 4
        return (high + low + 2 * close) / 4
    else:
        # (High + Low + Close) / 3 - HLC
        return (high + low + close) / 3

def simple_moving_average(candles, window_size):
    """Calculate SMA and return the calculation details"""
    if len(candles) < window_size:
        return {
            'value': None,
            'candles_used': candles,
            'reason': f"Insufficient data: {len(candles)} candles available, {window_size} required"
        }

    # Get the candles used in the calculation
    window_candles = candles[-window_size:]

    # Extract close prices
    closes = [float(c['close']) for c in window_candles]

    # Calculate SMA
    sma_value = sum(closes) / window_size

    # Return detailed info for verification
    return {
        'value': round(sma_value, 5),
        'candles_used': window_candles,
        'close_values': closes,
        'sum': sum(closes),
        'count': window_size,
        'formula': f"({' + '.join([str(c) for c in closes])}) / {window_size}"
    }

def efficiency_ratio(candles, period=10):
    """Calculate Kaufman Efficiency Ratio with calculation details"""
    if len(candles) < period:
        return {
            'value': None,
            'candles_used': candles,
            'reason': f"Insufficient data: {len(candles)} candles available, {period} required"
        }

    # Get candles for this calculation
    er_candles = candles[-period:]

    # Calculate weighted closes (HLC)
    wcloses = []
    for c in er_candles:
        wc = calculate_weighted_close(c['high'], c['low'], c['close'], kama_style=False)
        wcloses.append(wc)

    # Calculate price change (direction)
    price_change = abs(wcloses[-1] - wcloses[0])

    # Calculate volatility (path)
    volatility = 0
    volatility_steps = []
    for i in range(1, len(wcloses)):
        step = abs(wcloses[i] - wcloses[i-1])
        volatility += step
        volatility_steps.append(step)

    # Calculate ER
    if volatility == 0:
        er_value = 0
        formula = "0 (volatility is zero)"
    else:
        er_value = price_change / volatility
        formula = f"{price_change:.5f} / {volatility:.5f}"

    return {
        'value': round(er_value, 5),
        'candles_used': er_candles,
        'weighted_closes': wcloses,
        'price_change': price_change,
        'volatility': volatility,
        'volatility_steps': volatility_steps,
        'formula': formula
    }

def adaptive_moving_average(candles, er_value, fast_period=2, slow_period=30, kama_period=10, last_kama=None):
    """Calculate KAMA with calculation details"""
    if len(candles) < 1:
        return {
            'value': None,
            'reason': "No candles available"
        }

    # Use weighted close (H+L+Cx2)/4 for current price
    current_candle = candles[-1]
    current_price = calculate_weighted_close(
        current_candle['high'],
        current_candle['low'],
        current_candle['close'],
        kama_style=True
    )

    # Initialize KAMA if needed
    if last_kama is None:
        if len(candles) < kama_period:
            return {
                'value': current_price,
                'note': f"Initializing with current price, insufficient history for average ({len(candles)} < {kama_period})"
            }

        # Calculate initial KAMA as simple average
        kama_candles = candles[-kama_period:]
        weighted_closes = []
        for c in kama_candles:
            wc = calculate_weighted_close(c['high'], c['low'], c['close'], kama_style=True)
            weighted_closes.append(wc)

        last_kama = sum(weighted_closes) / len(weighted_closes)
        initialization = {
            'method': 'average',
            'candles_used': kama_candles,
            'weighted_closes': weighted_closes,
            'formula': f"({' + '.join([f'{wc:.5f}' for wc in weighted_closes])}) / {len(weighted_closes)}"
        }
    else:
        initialization = {
            'method': 'previous',
            'value': last_kama
        }

    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er_value * (fast_sc - slow_sc) + slow_sc) ** 2

    # Calculate KAMA
    kama_value = last_kama + sc * (current_price - last_kama)

    return {
        'value': round(kama_value, 5),
        'current_candle': current_candle,
        'current_weighted_price': current_price,
        'last_kama': last_kama,
        'initialization': initialization,
        'er_value': er_value,
        'fast_sc': fast_sc,
        'slow_sc': slow_sc,
        'sc': sc,
        'formula': f"{last_kama:.5f} + {sc:.5f} * ({current_price:.5f} - {last_kama:.5f})"
    }

def create_excel_friendly_validation(output_path, rows, headers=None):
    """Create a CSV file format optimized for Excel/Google Sheets validation"""
    with open(output_path, 'w', newline='') as csvfile:
        if headers:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
        else:
            # Get all possible headers from all rows
            all_headers = set()
            for row in rows:
                all_headers.update(row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=sorted(list(all_headers)))
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

def process_candle_data(candles_5m, candles_15m, candles_30m, analysis_date, end_date, output_directory):
    """Process candle data with maximum transparency for manual verification"""
    # Filter analysis window
    analysis_5m = candles_5m[(candles_5m['timestamp'] >= analysis_date) &
                             (candles_5m['timestamp'] < end_date)]

    print(f"Processing {len(analysis_5m)} 5M candles from {analysis_date} to {end_date}")

    # Create validation records for each calculation
    validation_records = []

    # Store previous KAMA values
    last_kama = {
        '5M': None,
        '15M': None,
        '30M': None
    }

    for _, row in analysis_5m.iterrows():
        current_time = row['timestamp']
        record = {
            'timestamp': current_time,
            'period_15m': (current_time.minute % 15) // 5,  # 0, 1, or 2
            'period_30m': (current_time.minute % 30) // 5,  # 0, 1, 2, 3, 4, or 5
        }

        # Add raw OHLC data for verification
        for col in ['open', 'high', 'low', 'close', 'volume']:
            record[f'5m_{col}'] = float(row[col])

        #--------------------------------------------------------
        # 5M SMA Calculation - Using 12 periods (1 hour)
        #--------------------------------------------------------
        # Get candles for SMA window
        window_5m_data = candles_5m[candles_5m['timestamp'] <= current_time].tail(12).to_dict('records')

        # Calculate SMA with details
        sma_5m_result = simple_moving_average(window_5m_data, 12)

        # Add basic SMA result
        record['sma_5m'] = sma_5m_result['value']

        # Add verification data
        record['sma_5m_candles_count'] = len(window_5m_data)
        if sma_5m_result['value'] is not None:
            record['sma_5m_sum'] = sma_5m_result['sum']
            record['sma_5m_avg'] = sma_5m_result['sum'] / 12 if len(window_5m_data) >= 12 else None

        # Add raw close values for manual verification
        for i, candle in enumerate(window_5m_data[-12:]):
            record[f'sma_5m_close_{i+1}'] = candle['close']

        #--------------------------------------------------------
        # 15M Timeframe Handling
        #--------------------------------------------------------
        # Get the most recent 20 completed 15M candles
        completed_15m = candles_15m[candles_15m['timestamp'] < current_time.floor('15min')].tail(20).to_dict('records')

        # Determine how to handle in-progress candle based on position in 15M window
        current_15m_period = record['period_15m']

        # Create the 15M in-progress candle appropriately
        if current_15m_period == 0:
            # For XX:00: Just use this single 5M candle
            inprogress_15m = {
                'timestamp': current_time.floor('15min'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            record['15m_candle_type'] = 'period_0_single_5m'

        elif current_15m_period == 1:
            # For XX:05: Combine XX:00 and XX:05 data
            period_start = current_time.floor('15min')
            candles_in_window = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                           (candles_5m['timestamp'] <= current_time)]

            if not candles_in_window.empty:
                inprogress_15m = {
                    'timestamp': period_start,
                    'open': float(candles_in_window.iloc[0]['open']),
                    'high': float(candles_in_window['high'].max()),
                    'low': float(candles_in_window['low'].min()),
                    'close': float(candles_in_window.iloc[-1]['close']),
                    'volume': float(candles_in_window['volume'].sum())
                }
                record['15m_candle_type'] = 'period_1_combined'

                # Add raw data for verification
                for i, c in enumerate(candles_in_window.to_dict('records')):
                    record[f'15m_inprogress_candle_{i}_time'] = c['timestamp']
                    record[f'15m_inprogress_candle_{i}_close'] = c['close']
            else:
                # Fallback - shouldn't happen
                inprogress_15m = None
                record['15m_candle_type'] = 'period_1_missing_data'

        elif current_15m_period == 2:
            # For XX:10: Use fully formed 15M candle from 5M data
            period_start = current_time.floor('15min')
            period_end = period_start + timedelta(minutes=15)

            # Get all 5M data for this 15M period, but don't exceed current time
            effective_end = min(period_end, current_time + timedelta(seconds=1))

            candles_in_window = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                           (candles_5m['timestamp'] < effective_end)]

            if not candles_in_window.empty:
                inprogress_15m = {
                    'timestamp': period_start,
                    'open': float(candles_in_window.iloc[0]['open']),
                    'high': float(candles_in_window['high'].max()),
                    'low': float(candles_in_window['low'].min()),
                    'close': float(candles_in_window.iloc[-1]['close']),
                    'volume': float(candles_in_window['volume'].sum())
                }
                record['15m_candle_type'] = 'period_2_full_15m'

                # Add raw data for verification
                record['15m_candle_candles_count'] = len(candles_in_window)
                for i, c in enumerate(candles_in_window.to_dict('records')):
                    record[f'15m_full_candle_{i}_time'] = c['timestamp']
                    record[f'15m_full_candle_{i}_close'] = c['close']
            else:
                # Fallback - shouldn't happen
                inprogress_15m = None
                record['15m_candle_type'] = 'period_2_missing_data'

        # Create 15M window for SMA calculation
        if inprogress_15m:
            window_15m = completed_15m + [inprogress_15m]

            # Add details about 15M candle for verification
            record['15m_inprogress_open'] = inprogress_15m['open']
            record['15m_inprogress_high'] = inprogress_15m['high']
            record['15m_inprogress_low'] = inprogress_15m['low']
            record['15m_inprogress_close'] = inprogress_15m['close']
        else:
            window_15m = completed_15m

        # Calculate 15M SMA
        sma_15m_result = simple_moving_average(window_15m, 20)
        record['sma_15m'] = sma_15m_result['value']

        # Add verification data
        record['sma_15m_candles_count'] = len(window_15m)
        if sma_15m_result['value'] is not None:
            record['sma_15m_sum'] = sma_15m_result['sum']
            record['sma_15m_avg'] = sma_15m_result['sum'] / 20 if len(window_15m) >= 20 else None

        # Add raw close values for manual verification
        for i, candle in enumerate(window_15m[-20:] if len(window_15m) >= 20 else window_15m):
            record[f'sma_15m_close_{i+1}'] = candle['close']

        #--------------------------------------------------------
        # 30M Timeframe Handling
        #--------------------------------------------------------
        # Similar pattern as 15M, but with 30M periods
        completed_30m = candles_30m[candles_30m['timestamp'] < current_time.floor('30min')].tail(30).to_dict('records')

        # For 30M, handle all 6 possible periods
        current_30m_period = record['period_30m']

        # Create the 30M in-progress candle based on position in 30M window
        if current_30m_period == 0:
            # For XX:00, XX:30: Just use this single 5M candle
            inprogress_30m = {
                'timestamp': current_time.floor('30min'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            record['30m_candle_type'] = 'period_0_single_5m'
        else:
            # For all other periods: Combine data from start of period to current time
            period_start = current_time.floor('30min')
            candles_in_window = candles_5m[(candles_5m['timestamp'] >= period_start) &
                                           (candles_5m['timestamp'] <= current_time)]

            if not candles_in_window.empty:
                inprogress_30m = {
                    'timestamp': period_start,
                    'open': float(candles_in_window.iloc[0]['open']),
                    'high': float(candles_in_window['high'].max()),
                    'low': float(candles_in_window['low'].min()),
                    'close': float(candles_in_window.iloc[-1]['close']),
                    'volume': float(candles_in_window['volume'].sum())
                }
                record['30m_candle_type'] = f'period_{current_30m_period}_combined'

                # Add raw data for validation (first few and last few candles)
                record['30m_inprogress_candles_count'] = len(candles_in_window)

                # Get first 3 and last 3 candles (if available) for verification
                head_candles = candles_in_window.head(3).to_dict('records')
                tail_candles = candles_in_window.tail(3).to_dict('records')

                for i, c in enumerate(head_candles):
                    record[f'30m_inprogress_first_{i+1}_time'] = c['timestamp']
                    record[f'30m_inprogress_first_{i+1}_close'] = c['close']

                for i, c in enumerate(tail_candles):
                    record[f'30m_inprogress_last_{i+1}_time'] = c['timestamp']
                    record[f'30m_inprogress_last_{i+1}_close'] = c['close']
            else:
                # Fallback - shouldn't happen
                inprogress_30m = None
                record['30m_candle_type'] = f'period_{current_30m_period}_missing_data'

        # Special case for period 5 (XX:25, XX:55) - check if this is the final candle of the 30M
        if current_30m_period == 5:
            # This is the final candle of the 30M window
            record['30m_candle_note'] = 'Final candle of 30M window'

        # Create 30M window for SMA calculation
        if inprogress_30m:
            window_30m = completed_30m + [inprogress_30m]

            # Add details about 30M candle for verification
            record['30m_inprogress_open'] = inprogress_30m['open']
            record['30m_inprogress_high'] = inprogress_30m['high']
            record['30m_inprogress_low'] = inprogress_30m['low']
            record['30m_inprogress_close'] = inprogress_30m['close']
        else:
            window_30m = completed_30m

        # Calculate 30M SMA
        sma_30m_result = simple_moving_average(window_30m, 30)
        record['sma_30m'] = sma_30m_result['value']

        # Add verification data
        record['sma_30m_candles_count'] = len(window_30m)
        if sma_30m_result['value'] is not None:
            record['sma_30m_sum'] = sma_30m_result['sum']
            record['sma_30m_avg'] = sma_30m_result['sum'] / 30 if len(window_30m) >= 30 else None

        # Add raw close values for verification (first few and last few only to save space)
        if len(window_30m) >= 30:
            first_five = window_30m[-30:][:5]
            last_five = window_30m[-30:][-5:]
        else:
            first_five = window_30m[:5] if len(window_30m) >= 5 else window_30m
            last_five = window_30m[-5:] if len(window_30m) >= 5 else window_30m

        for i, candle in enumerate(first_five):
            record[f'sma_30m_first_{i+1}_close'] = candle['close']

        for i, candle in enumerate(last_five):
            record[f'sma_30m_last_{i+1}_close'] = candle['close']

        #--------------------------------------------------------
        # Kaufman Indicators for 5M Timeframe
        #--------------------------------------------------------
        # Get candles for 5M history (use last 30 for calculations)
        history_5m = candles_5m[candles_5m['timestamp'] <= current_time].tail(30).to_dict('records')

        # Calculate Efficiency Ratio for 5M
        er_5m_result = efficiency_ratio(history_5m, period=10)
        record['ker_5m'] = er_5m_result['value']

        # Add verification data
        if er_5m_result['value'] is not None:
            record['ker_5m_price_change'] = er_5m_result['price_change']
            record['ker_5m_volatility'] = er_5m_result['volatility']

            # Add raw weighted close values
            for i, wc in enumerate(er_5m_result['weighted_closes']):
                record[f'ker_5m_wclose_{i+1}'] = wc

        # Calculate KAMA for 5M
        if er_5m_result['value'] is not None:
            kama_5m_result = adaptive_moving_average(
                history_5m,
                er_5m_result['value'],
                last_kama=last_kama['5M']
            )
            record['kama_5m'] = kama_5m_result['value']

            # Add verification data
            record['kama_5m_sc'] = kama_5m_result['sc']

            if 'current_weighted_price' in kama_5m_result:
                record['kama_5m_current_price'] = kama_5m_result['current_weighted_price']

            if 'last_kama' in kama_5m_result:
                record['kama_5m_previous'] = kama_5m_result['last_kama']

            # Update last KAMA for next iteration
            last_kama['5M'] = kama_5m_result['value']

        #--------------------------------------------------------
        # Kaufman Indicators for 15M Timeframe
        #--------------------------------------------------------
        # Create 15M history by combining completed candles and current in-progress
        history_15m = completed_15m
        if inprogress_15m:
            history_15m = history_15m + [inprogress_15m]

        # Calculate Efficiency Ratio for 15M
        er_15m_result = efficiency_ratio(history_15m, period=10)
        record['ker_15m'] = er_15m_result['value']

        # Add verification data
        if er_15m_result['value'] is not None:
            record['ker_15m_price_change'] = er_15m_result['price_change']
            record['ker_15m_volatility'] = er_15m_result['volatility']

            # Add raw weighted close values (first 3 and last 3)
            if len(er_15m_result['weighted_closes']) >= 6:
                first_three = er_15m_result['weighted_closes'][:3]
                last_three = er_15m_result['weighted_closes'][-3:]
            else:
                first_three = er_15m_result['weighted_closes']
                last_three = er_15m_result['weighted_closes']

            for i, wc in enumerate(first_three):
                record[f'ker_15m_first_wclose_{i+1}'] = wc

            for i, wc in enumerate(last_three):
                record[f'ker_15m_last_wclose_{i+1}'] = wc

        # Calculate KAMA for 15M
        if er_15m_result['value'] is not None:
            kama_15m_result = adaptive_moving_average(
                history_15m,
                er_15m_result['value'],
                last_kama=last_kama['15M']
            )
            record['kama_15m'] = kama_15m_result['value']

            # Add verification data
            record['kama_15m_sc'] = kama_15m_result['sc']

            if 'current_weighted_price' in kama_15m_result:
                record['kama_15m_current_price'] = kama_15m_result['current_weighted_price']

            if 'last_kama' in kama_15m_result:
                record['kama_15m_previous'] = kama_15m_result['last_kama']

            # Update last KAMA for next iteration
            last_kama['15M'] = kama_15m_result['value']

        #--------------------------------------------------------
        # Kaufman Indicators for 30M Timeframe
        #--------------------------------------------------------
        # Similar pattern to 15M
        history_30m = completed_30m
        if inprogress_30m:
            history_30m = history_30m + [inprogress_30m]

        # Calculate Efficiency Ratio for 30M
        er_30m_result = efficiency_ratio(history_30m, period=10)
        record['ker_30m'] = er_30m_result['value']

        # Add verification data
        if er_30m_result['value'] is not None:
            record['ker_30m_price_change'] = er_30m_result['price_change']
            record['ker_30m_volatility'] = er_30m_result['volatility']

            # Add raw weighted close values (first 3 and last 3)
            if len(er_30m_result['weighted_closes']) >= 6:
                first_three = er_30m_result['weighted_closes'][:3]
                last_three = er_30m_result['weighted_closes'][-3:]
            else:
                first_three = er_30m_result['weighted_closes']
                last_three = er_30m_result['weighted_closes']

            for i, wc in enumerate(first_three):
                record[f'ker_30m_first_wclose_{i+1}'] = wc

            for i, wc in enumerate(last_three):
                record[f'ker_30m_last_wclose_{i+1}'] = wc

        # Calculate KAMA for 30M
        if er_30m_result['value'] is not None:
            kama_30m_result = adaptive_moving_average(
                history_30m,
                er_30m_result['value'],
                last_kama=last_kama['30M']
            )
            record['kama_30m'] = kama_30m_result['value']

            # Add verification data
            record['kama_30m_sc'] = kama_30m_result['sc']

            if 'current_weighted_price' in kama_30m_result:
                record['kama_30m_current_price'] = kama_30m_result['current_weighted_price']

            if 'last_kama' in kama_30m_result:
                record['kama_30m_previous'] = kama_30m_result['last_kama']

            # Update last KAMA for next iteration
            last_kama['30M'] = kama_30m_result['value']

        # Add the record to validation records
        validation_records.append(record)

    # Create Excel-friendly output files
    validation_file = os.path.join(output_directory, 'Complete_Verification_Data.csv')
    simplified_file = os.path.join(output_directory, 'Simplified_Results.csv')

    # Create the full validation file
    create_excel_friendly_validation(validation_file, validation_records)

    # Create a simplified results file with just the main indicators
    simple_fields = ['timestamp', 'period_15m', 'period_30m',
                     'sma_5m', 'sma_15m', 'sma_30m',
                     'ker_5m', 'ker_15m', 'ker_30m',
                     'kama_5m', 'kama_15m', 'kama_30m']

    simplified_records = []
    for record in validation_records:
        simple_record = {field: record.get(field) for field in simple_fields}
        simplified_records.append(simple_record)

    create_excel_friendly_validation(simplified_file, simplified_records, simple_fields)

    print(f"Full verification data saved to: {validation_file}")
    print(f"Simplified results saved to: {simplified_file}")

    return validation_records

def main():
    directory = r'D:\\'  # Update this path to your data directory

    try:
        # Load data
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
        print(candles_5m[candles_5m['timestamp'] < analysis_date].tail(24))

        print("\nPrior 5 Hours 15M Data:")
        print(candles_15m[candles_15m['timestamp'] < analysis_date].tail(20))

        print("\nPrior 15 Hours 30M Data:")
        print(candles_30m[candles_30m['timestamp'] < analysis_date].tail(30))

        # Process data with maximum transparency
        print("\nProcessing indicators with transparent calculations...")
        results = process_candle_data(candles_5m, candles_15m, candles_30m,
                                      analysis_date, end_date, directory)

        # Basic data validation
        print("\nData Validation Summary:")
        print(f"Total records processed: {len(results)}")

        # Check for consistency - each timeframe period should have constant values
        print("\nVerifying SMA consistency by timeframe period...")

        # Check 15M periods
        print("\n15M SMA Consistency Check:")
        for period in [0, 1, 2]:
            period_results = [r for r in results if r['period_15m'] == period]
            sma_values = set([r['sma_15m'] for r in period_results if r['sma_15m'] is not None])
            print(f"Period {period}: {len(sma_values)} unique values")
            for value in sma_values:
                print(f"  Value: {value}")
                timestamps = [r['timestamp'] for r in period_results if r['sma_15m'] == value]
                print(f"  Used in {len(timestamps)} timestamps, e.g.: {timestamps[:2]}")

        # Check 30M periods
        print("\n30M SMA Consistency Check:")
        for period in range(6):
            period_results = [r for r in results if r['period_30m'] == period]
            sma_values = set([r['sma_30m'] for r in period_results if r['sma_30m'] is not None])
            print(f"Period {period}: {len(sma_values)} unique values")
            for value in sma_values:
                print(f"  Value: {value}")
                timestamps = [r['timestamp'] for r in period_results if r['sma_30m'] == value]
                print(f"  Used in {len(timestamps)} timestamps, e.g.: {timestamps[:2]}")

        print("\nValidation and verification complete!")
        print("Files have been created with all raw calculation data.")
        print("You can now import them into Google Sheets for manual verification.")

    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
