import pandas as pd
import numpy as np

# Load the CSV files
comprehensive_data_path = '.../Maneuver.csv'
angular_velocity_data_path = '.../angular_velocity_data.csv'

comprehensive_df = pd.read_csv(comprehensive_data_path)
angular_velocity_df = pd.read_csv(angular_velocity_data_path)

# Ensure the dataframes are aligned by index
assert len(comprehensive_df) == len(angular_velocity_df), "DataFrame lengths do not match."

# Define the new maneuver classification logic
# Threshold from UtsoThresholding.py
Threshold = 0.15326725726481527

def classify_maneuver(angular_velocities):
    if np.any(np.abs(angular_velocities) > Threshold):
        if np.mean(angular_velocities) < 0:
            return "Turn Right"
        else:
            return "Turn Left"
    return "Straight"

# Apply the classification logic using a rolling window
window_size = 20
new_maneuver_types = []

for i in range(len(angular_velocity_df)):
    window_start = max(i - window_size + 1, 0)
    window = angular_velocity_df['angular_velocity_z'][window_start:i+1]
    maneuver_type = classify_maneuver(window)
    new_maneuver_types.append(maneuver_type)

# Update the comprehensive DataFrame with the new maneuver types
comprehensive_df['maneuver_type'] = new_maneuver_types

# Save the updated DataFrame to a new CSV file
updated_maneuver_data_path = '.../Maneuver.csv'
comprehensive_df.to_csv(updated_maneuver_data_path, index=False)

updated_maneuver_data_path
