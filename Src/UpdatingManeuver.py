import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# Load the angular_velocity and maneuver data
angular_velocity_file = '.../angular_velocity.csv'
maneuver_file = '.../comprehensive_data.csv'

df_angular_velocity = pd.read_csv(angular_velocity_file)
df_maneuver = pd.read_csv(maneuver_file)

# Process angular_velocity data to get Euler angles and yaw gradient
euler_angles = df_angular_velocity.apply(lambda row: R.from_quat([row['orientation_x'], row['orientation_y'], row['orientation_z'], row['orientation_w']]).as_euler('xyz', degrees=True), axis=1)
df_angular_velocity['yaw_gradient'] = np.gradient([yaw for _, _, yaw in euler_angles])

# Classify maneuver type
def classify_maneuver(index, window_size=10):
    if index + window_size > len(df_odometry):
        return "Straight"
    gradients = df_angular_velocity['yaw_gradient'][index:index+window_size]
    if np.any(np.abs(gradients) > 1.2):
        return "Turn Left" if np.mean(gradients) > 0 else "Turn Right"
    return "Straight"

# Update the maneuver_type column
df_maneuver['maneuver_type'] = [classify_maneuver(i) for i in df_maneuver.index]

# Save the updated DataFrame
df_maneuver.to_csv(maneuver_file, index=False)
