import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load the odometry and comprehensive data
odometry_file = '/home/nrc/Documents/Maneuver/Modified code/Comprehensive4/odometry_data.csv'
comprehensive_file = '/home/nrc/Documents/Maneuver/Modified code/Comprehensive4/comprehensive_data.csv'

df_odometry = pd.read_csv(odometry_file)
df_comprehensive = pd.read_csv(comprehensive_file)

# Process odometry data to get Euler angles and yaw gradient
euler_angles = df_odometry.apply(lambda row: R.from_quat([row['orientation_x'], row['orientation_y'], row['orientation_z'], row['orientation_w']]).as_euler('xyz', degrees=True), axis=1)
df_odometry['yaw_gradient'] = np.gradient([yaw for _, _, yaw in euler_angles])

# Classify maneuver type
def classify_maneuver(index, window_size=10):
    if index + window_size > len(df_odometry):
        return "Straight"
    gradients = df_odometry['yaw_gradient'][index:index+window_size]
    if np.any(np.abs(gradients) > 1.2):
        return "Turn Left" if np.mean(gradients) > 0 else "Turn Right"
    return "Straight"

# Update the maneuver_type column
df_comprehensive['maneuver_type'] = [classify_maneuver(i) for i in df_comprehensive.index]

# Save the updated DataFrame
df_comprehensive.to_csv(comprehensive_file, index=False)
