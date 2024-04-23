import cv2
import numpy as np
import os
import csv
import re
from cv2 import aruco

# Define paths and parameters
frames_directory = ".../steering_images"
output_csv_path = ".../steering_angles.csv"
camera_matrix = np.array([[532.772603, 0.000000, 315.129715],
                          [0.000000, 530.554625, 231.588145],
                          [0.000000, 0.000000, 1.000000]])
distortion_coeffs = np.array([0.064885, -0.176797, 0.002945, -0.001015, 0.000000])
marker_length = 0.067  # In meters

# Custom function to sort frame filenames numerically
def sort_frames_numerically(file_name):
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else 0

# Function to calculate yaw angle from rotation vector
def calculate_yaw_angle_from_rvec(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(yaw)

# Function to update and track total rotation
previous_yaw = None
total_rotation = 0

def update_total_rotation(current_yaw):
    global previous_yaw, total_rotation
    if previous_yaw is not None:
        rotation_difference = previous_yaw - current_yaw
        if rotation_difference > 180:
            rotation_difference -= 360
        elif rotation_difference < -180:
            rotation_difference += 360
        total_rotation -= rotation_difference
    previous_yaw = current_yaw

# Main processing function
def process_frames_and_write_to_csv():
    global total_rotation  # Make sure to use the global variable
    frame_files = sorted(os.listdir(frames_directory), key=sort_frames_numerically)
    frame_files = [file for file in frame_files if file.endswith(".png")]
    
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Frame', 'Current Yaw (degrees)', 'Total Rotation (degrees)', 'Marker Detected'])

        for frame_file in frame_files:
            frame_path = os.path.join(frames_directory, frame_file)
            image = cv2.imread(frame_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict)
            
            if ids is not None and len(ids) > 0:
                rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion_coeffs)
                current_yaw = calculate_yaw_angle_from_rvec(rvecs[0])
                update_total_rotation(current_yaw)  # This updates the global total_rotation
                marker_detected = 'Yes'
            else:
                current_yaw = 'N/A'  # This needs to be reset for each frame
                marker_detected = 'No'
            
            # Ensure 'total_rotation' has a value even if the marker is not detected
            total_rotation_value = (-1) * total_rotation if marker_detected == 'Yes' else 'N/A'
            
            # Write the current frame's data to the CSV file
            csv_writer.writerow([frame_file, current_yaw, total_rotation_value, marker_detected])


# Execute the main function
process_frames_and_write_to_csv()
