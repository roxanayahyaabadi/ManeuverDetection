import pandas as pd

# Load the combined CSV file with all samples, including final maneuver types and confidence scores
csv_path = '.../all_samples_with_final_labels.csv'
data = pd.read_csv(csv_path)

# Specify the number of frames with the lowest confidence scores you want to list
n = 3000  # You can change this number

# Filter to include only test set data
test_data = data[data['Data_Type'] == 'Test']

# Sort the test data by confidence score in ascending order
sorted_test_data_by_confidence = test_data.sort_values(by='max_proba', ascending=True)

# Select the top n frames with the lowest confidence scores
lowest_confidence_frames = sorted_test_data_by_confidence.head(n)

# Sort these selected frames by frame_id in numeric order
lowest_confidence_frames_sorted = lowest_confidence_frames.sort_values(by='frame_id', ascending=True)[['frame_id', 'max_proba']]

# Save these frames to a CSV file
output_csv_path = '.../lowest_confidence_iter_1.csv'
lowest_confidence_frames_sorted.to_csv(output_csv_path, index=False)

print(f"The {n} frames with the lowest confidence scores, sorted by frame_id, have been saved to {output_csv_path}.")
