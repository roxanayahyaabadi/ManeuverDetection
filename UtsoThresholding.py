# Re-importing necessary libraries after code execution state reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Re-load the dataset (assuming the path needs to be re-defined)
velocity_angles_df = pd.read_csv('.../angular_velocity_data.csv')

# Define the custom Otsu's method function
def otsu_threshold_method(data):
    # Compute the histogram of the data

    clean_data = data[~np.isnan(data)]

    bin_counts, bin_edges = np.histogram(clean_data, bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialize maximum sigma_b_squared (between-class variance) and optimal threshold
    max_sigma_b_squared = 0
    optimal_threshold = bin_centers[0]
    
    for i, bin_center in enumerate(bin_centers):
        # Divide data into two groups: below and above the current threshold
        weight1 = np.sum(bin_counts[:i+1])
        weight2 = np.sum(bin_counts[i+1:])
        mean1 = np.sum(bin_centers[:i+1] * bin_counts[:i+1]) / weight1 if weight1 > 0 else 0
        mean2 = np.sum(bin_centers[i+1:] * bin_counts[i+1:]) / weight2 if weight2 > 0 else 0
        
        # Calculate the between-class variance (sigma_b_squared)
        sigma_b_squared = weight1 * weight2 * (mean1 - mean2) ** 2
        
        # Update optimal threshold if current sigma_b_squared is larger
        if sigma_b_squared > max_sigma_b_squared:
            max_sigma_b_squared = sigma_b_squared
            optimal_threshold = bin_center
    
    # Compute the histogram of the data
    bin_counts, bin_edges = np.histogram(clean_data, bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialize variables to store the maximum between-class variance and the optimal threshold
    max_sigma_b_squared = 0
    optimal_threshold = bin_centers[0]
    
    # Iterate through all possible thresholds (bin centers)
    for i, bin_center in enumerate(bin_centers):
        # Split the histogram into two parts at the current threshold
        weight1 = np.sum(bin_counts[:i+1])
        weight2 = np.sum(bin_counts[i+1:])
        mean1 = np.sum(bin_centers[:i+1] * bin_counts[:i+1]) / weight1 if weight1 > 0 else 0
        mean2 = np.sum(bin_centers[i+1:] * bin_counts[i+1:]) / weight2 if weight2 > 0 else 0
        
        # Calculate the between-class variance
        sigma_b_squared = weight1 * weight2 * (mean1 - mean2) ** 2
        
        # Update the maximum between-class variance and optimal threshold if necessary
        if sigma_b_squared > max_sigma_b_squared:
            max_sigma_b_squared = sigma_b_squared
            optimal_threshold = bin_center

    plt.rc('font', size=20) 
    # Plot the histogram and the optimal threshold
    plt.figure(figsize=(10, 6))
    plt.hist(clean_data, bins=256, alpha=0.7, label='Angular Velocity Z')
    plt.axvline(optimal_threshold, color='r', linestyle='dashed', linewidth=3, label='Optimal Threshold')
    plt.title('Histogram of Angular Velocity Z with Optimal Threshold')
    plt.xlabel('Angular Velocity Z (rad/s)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return optimal_threshold


# Calculate the optimal threshold using Otsu's method and plot the histogram
optimal_threshold = otsu_threshold_method(velocity_angles_df['angular_velocity_z'].values)
print(optimal_threshold)
