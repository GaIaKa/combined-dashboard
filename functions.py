import pandas as pd


def removenan(b):
    b = b.copy()
    b['timestamp'] = pd.to_datetime(b['timestamp'])

    # Create a new column for the minute without seconds
    b.loc[:, 'minute'] = b['timestamp'].dt.floor('T')

    # Full range of expected minutes between min and max timestamp
    full_range = pd.date_range(start=b['minute'].min(), end=b['minute'].max(), freq='T')

    # Find the missing minutes
    missing_minutes = full_range.difference(b['minute'])

    # Initialize a list to store missing timestamps
    missing_timestamps = []

    # Initialize a counter for missing minutes
    missing_minute_count = 0

    # Loop through the missing minutes and find the ones that should be added
    for missing_minute in missing_minutes:
        prev_time = missing_minute - pd.Timedelta(minutes=1)
        next_time = missing_minute + pd.Timedelta(minutes=1)

        has_prev = b['timestamp'].dt.floor('T').eq(prev_time).any()
        has_next = b['timestamp'].dt.floor('T').eq(next_time).any()

        if has_prev and has_next:
            prev_seconds = b.loc[b['timestamp'].dt.floor('T') == prev_time, 'timestamp'].dt.second.values
            next_seconds = b.loc[b['timestamp'].dt.floor('T') == next_time, 'timestamp'].dt.second.values

            #if 59 in prev_seconds and 0 in next_seconds:
            missing_timestamps.append(missing_minute)
            missing_minute_count += 1  # Increment the counter for each missing minute

    # Add the missing minutes to the DataFrame
    for missing_time in missing_timestamps:
        new_row = {'timestamp': missing_time, 'Efield': None, 'minute': missing_time}
        b = pd.concat([b[b['timestamp'] < missing_time], 
                       pd.DataFrame([new_row]), 
                       b[b['timestamp'] >= missing_time]]).reset_index(drop=True)

    # Shift data to fill in the added rows
    for missing_time in missing_timestamps:
        missing_pos = b[b['timestamp'] == missing_time].index[0]

        if missing_pos + 1 < len(b):
            b.loc[missing_pos, 'Efield'] = b.loc[missing_pos + 1, 'Efield']
            b.loc[missing_pos, 'minute'] = b.loc[missing_pos + 1, 'minute']

            b.loc[missing_pos + 1:, 'Efield'] = b.loc[missing_pos:, 'Efield'].shift(-1)
            b.loc[missing_pos + 1:, 'minute'] = b.loc[missing_pos:, 'minute'].shift(-1)

    # Drop rows where Efield is still NaN
    b = b.dropna(subset=['Efield']).reset_index(drop=True)

    # Print the number of missing minutes
    print(f"Number of missing minutes added: {missing_minute_count}")

    return b