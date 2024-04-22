
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load the ground truth data
ground_truth_path = '.../Ground_truth.csv'
ground_truth = pd.read_csv(ground_truth_path)

# Load the predictions data
# predictions_path = '.../all_samples_with_final_labels.csv'  # Update with the actual path
predictions_path = '.../all_samples_with_final_labels.csv'
predictions = pd.read_csv(predictions_path)

# Merge the ground truth and predictions based on frame_id
combined = ground_truth.merge(predictions, on='frame_id', how='left')

# Filter for test data
test_data = combined[combined['Data_Type'] == 'Test']

# Ensure the ground truth and predictions are in the same format
y_true = test_data['Ground truth']  # Update column name as per Ground_truth.csv
y_pred = test_data['Final_Maneuver_Type']

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')  # 'weighted' accounts for label imbalance
recall = recall_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

# Print the metrics
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precision:", precision)
print("Confusion Matrix:\n", conf_matrix)


# Define the classes and confusion matrix
classes = ["Left lane change", "Right lane change", "Straight", "Turn Left", "Turn Right"]

row_sums = conf_matrix.sum(axis=1)
row_sums[row_sums == 0] = 1  # Prevent division by zero

# Calculate percentages
confusion_matrix_percent = conf_matrix.astype('float') / row_sums[:, np.newaxis] * 100

# Replace NaN values with 0 (if any)
confusion_matrix_percent = np.nan_to_num(confusion_matrix_percent)
# Set the context to "talk" for larger fonts
sns.set_context("talk", font_scale=1.25)
# Create a heatmap from the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix_percent, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 25})

# Rotate labels for clarity
plt.xticks(rotation=15)
plt.yticks(rotation=45)

# Adding heatmap decorations
plt.title("Confusion Matrix Heatmap for RF-Second iteration (Percentages)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

#Relabeling the samples with the lowest confidence score
feature_vector_path = '.../feature_vector_updated_iter1.csv'
low_confidence_path = '.../low_confidence_samples_with_ground_truth.csv'

feature_vector_df = pd.read_csv(feature_vector_path)
low_confidence_df = pd.read_csv(low_confidence_path)

# Merge the feature_vector_df with low_confidence_df to get the Ground truth values
merged_df = feature_vector_df.merge(low_confidence_df[['frame_id', 'Ground truth']],
                                    on='frame_id',
                                    how='left')

# Update 'maneuver_type' in feature_vector_df with 'Ground truth' from low_confidence_df where available
merged_df.loc[merged_df['Ground truth'].notnull(), 'maneuver_type'] = merged_df['Ground truth']

# Now we have 'Ground truth' as a separate column, which we don't need anymore. We can drop it.
final_df = merged_df.drop(columns=['Ground truth'])


# Save the updated DataFrame back to a CSV file
final_df.to_csv(feature_vector_path, index=False)

print("The feature-vector CSV file has been updated with ground truth values and saved.")
