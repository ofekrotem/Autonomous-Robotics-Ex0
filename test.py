import pandas as pd
import numpy as np

# Constants
LIGHTSPEED = 299792458  # Speed of light in m/s

def parse_raw_line(line):
    """Parse a raw measurement line and return a formatted dictionary."""
    # Split the line into components
    parts = line.strip().split(',')
    # Create a dictionary with placeholder values
    data = {
        'GPS_Time': np.nan,
        'Svid': np.nan,
        'Sat.X': np.nan,
        'Sat.Y': np.nan,
        'Sat.Z': np.nan,
        'Pseudo-Range': np.nan,
        'Cn0DbHz': np.nan,
        'PseudorangeRateMetersPerSecond': np.nan
    }
    # Try to assign actual values from parts, handle missing or incomplete data
    try:
        data['GPS_Time'] = float(parts[1]) + float(parts[5])  # TimeNanos + FullBiasNanos
        data['Svid'] = parts[11]
        data['Cn0DbHz'] = float(parts[16])
        data['PseudorangeRateMetersPerSecond'] = float(parts[17])
        # Pseudo-Range simplification (not actual calculation)
        data['Pseudo-Range'] = float(parts[14]) / LIGHTSPEED
    except IndexError:
        # Handle missing data or incorrect format
        print(f"Error parsing line: {line}")
        pass
    return data

# File path
data_path = './data/gnss_log_2024_04_13_19_53_33.txt'

# Read data
processed_data = []
with open(data_path, 'r') as file:
    for line in file:
        if line.startswith('Raw'):  # Assuming 'Raw' lines contain the data we're interested in
            parsed_data = parse_raw_line(line)
            processed_data.append(parsed_data)

# Convert to DataFrame
df = pd.DataFrame(processed_data)

# Optionally fill NaNs if you have default values or need to clean up
# df.fillna(method='ffill', inplace=True)  # Forward fill or use 'bfill' or specific values

# Save to CSV
df.to_csv('processed_gnss_data.csv', index=False)

print("Data processing complete, output saved to 'processed_gnss_data.csv'.")
