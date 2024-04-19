import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import mode
import numpy as np

def apply_majority_voting_windowing(series, window_size=10):
    """Applies majority voting to a pandas series within a moving window."""
    smoothed = series.rolling(window=window_size, min_periods=1, center=True).apply(lambda x: mode(x)[0], raw=False)
    return smoothed.astype(int)

# Load the CSV file
file_path = 'C:/Users/roxan/OneDrive - The University of Western Ontario/ITSC/driver_20/MLP/Fourth/feature_vector_iter4.csv'
data = pd.read_csv(file_path)

# Preprocessing
data['maneuver_type_encoded'] = pd.Series(dtype='float')
data = data.dropna(subset=['Total Rotation (degrees)'])
has_label_mask = data['maneuver_type'].notna()
label_encoder = LabelEncoder()
data.loc[has_label_mask, 'maneuver_type_encoded'] = label_encoder.fit_transform(data.loc[has_label_mask, 'maneuver_type'])

# Separate labeled and unlabeled data
labeled_data = data[has_label_mask].copy()
unlabeled_data = data[~has_label_mask].copy()

# Define feature columns
feature_columns = [
    'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
    'latitude', 'longitude', 'altitude', 'delta_x', 'delta_y', 'delta_z', 'velocity_x', 'velocity_y', 'velocity_z',
    'delta_velocity_x', 'delta_velocity_y', 'delta_velocity_z', 'Total Rotation (degrees)'
]

X_train = labeled_data[feature_columns]
y_train = labeled_data['maneuver_type_encoded'].astype(int)
X_test = unlabeled_data[feature_columns]

# Addressing class imbalance by oversampling
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Initialize and train the MLP model with two layers: 50 and 20 neurons
model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict on test data
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
unlabeled_data['predicted_label'] = y_pred

# Add probabilities to the test data
for i, class_name in enumerate(label_encoder.classes_):
    unlabeled_data[f'prob_{class_name}'] = proba[:, i]
unlabeled_data['max_proba'] = proba.max(axis=1)

# Apply windowing to smooth predictions
window_size = 10
unlabeled_data['smoothed_label'] = apply_majority_voting_windowing(unlabeled_data['predicted_label'], window_size)

max_proba = unlabeled_data['max_proba'].values
threshold = np.percentile(max_proba, 25)  # Adjust this based on histogram analysis
print(f"Automatically determined threshold: {threshold:.3f}")

# Identify low and high confidence samples
low_confidence_samples = unlabeled_data[unlabeled_data['max_proba'] < threshold]
high_confidence_samples = unlabeled_data[unlabeled_data['max_proba'] >= threshold]

# Mark data as train or test, and prepare for combination
labeled_data['Data_Type'] = 'Train'
unlabeled_data['Data_Type'] = 'Test'

# For labeled_data, assign NaN to max_proba since it's actual data, not predicted
labeled_data['max_proba'] = pd.NA  # Using pd.NA for compatibility with different data types

# Combine labeled and unlabeled data
combined_data = pd.concat([labeled_data, unlabeled_data])

# Ensure the data is sorted by frame_id
combined_data.sort_values(by='frame_id', inplace=True)

default_class_name = 'Straight'
default_class_index = label_encoder.transform([default_class_name])[0]

# Determine the final maneuver type
combined_data['Final_Maneuver_Type'] = combined_data.apply(
    lambda x: x['maneuver_type'] if pd.notna(x['maneuver_type'])
    else label_encoder.inverse_transform([int(x['smoothed_label'])] if pd.notna(x['smoothed_label']) else [default_class_index])[0], axis=1)


## Save CSV files
# output_folder = 'C:/Users/roxan/OneDrive - The University of Western Ontario/ITSC/Code/Windowing/driver_17/Third_iteration/'
output_folder = 'C:/Users/roxan/OneDrive - The University of Western Ontario/ITSC/driver_20/MLP/Fourth/'

unlabeled_data.to_csv(f'{output_folder}test_samples_with_probabilities.csv', index=False)
low_confidence_samples.to_csv(f'{output_folder}low_confidence_samples.csv', index=False)
high_confidence_samples.to_csv(f'{output_folder}high_confidence_samples.csv', index=False)
output_file_path = f'{output_folder}all_samples_with_final_labels.csv'
combined_data[['frame_id', 'Data_Type', 'Final_Maneuver_Type', 'max_proba']].to_csv(output_file_path, index=False)

print("CSV files for all samples, including final maneuver types, have been saved.")

