# Driving Maneuver Detection

This repository hosts implementations for the "From Wheels to AI: Advanced Driving Maneuver Detection Through Active Learning Employing Vehicle Dynamics" submitted paper, detecting driving maneuvers using rule-based and machine-learning approaches. It aims to enhance vehicle safety and intelligence by accurately identifying driving behaviors, combining traditional and modern techniques for comprehensive analysis.

![](Media/ManeuverDetection.gif)

# Abstract

Detecting driving maneuvers from vehicle dynamics and other driver-specific data can improve the safety margins of Advanced Driver Assistance Systems (ADAS) by enabling early detection of dangerous driving events. Building these predictive models requires large annotated datasets that are labor-intensive to construct using manual labeling. To overcome this challenge, this paper introduces a novel and effective approach for maneuver detection by harnessing vehicle dynamic signals through Active Learning. By engineering informative dynamic signals as features and employing an Active Learning strategy, maneuver types can be accurately identified. Utilizing XGBoost as its core, our approach achieved a Macro accuracy of 91.40\%—a critical metric for assessing performance in imbalanced datasets— across more than 44,200 test samples from a total of 58,936.

# Dynamic Signals

We proposed to use some informative dynamic signals to detect the maneuver. 
The dataset used in this project can be found in the `Src/Feature_Vector_initial.csv` file.

# Rule-based Maneuver Detection

In this approach, we explore various vehicle signals alongside the steering wheel's position, which is derived from images of the steering wheel. Our investigation identifies the "Angular Velocity Z component" as the most informative feature. `Src/OtsuThresholding.py` shows how Otsu's thresholding, calculates the optimal threshold. Then, `Rule-Based_Maneuver.py` shows how to apply Utso's thresholding on the angular velocity Z component to segment the maneuvers.  

![](Media/Utso.png)

# Active Learning-based Maneuver Detection

In the second approach, due to the limitations of the rule-based method in handling five maneuvers while focusing on only one feature, which is insufficient for lane-changing maneuvers, we propose a novel approach based on "Active Learning" that leverages ML as its core building block.  Here we proposed a methodology that can handle a huge number of samples by keeping the human in the loop as the oracle while the majority of samples are labeled by the Machine learning model iteratively. 

![](Media/AL.png)


# Steering Wheel Angle Detection

The `Src/SteeringwheelDetection.csv` file shows how to detect the relative Steering Wheel angle using decoding the ArUco marker stuck on the steering wheel. To estimate the steering wheel's rotation angle, an ArUco marker attached to the wheel is monitored through a sequence of images. By using calibrated camera parameters, the 3D pose of the marker is determined, and rotation vectors are converted into yaw angles to assess the wheel’s orientation. Changes in yaw angles between consecutive frames are used to calculate the cumulative rotation from the starting position. This process also includes meticulous tracking of each frame's yaw angle and the marker's visibility status. If the marker is obscured by a hand, the rotation angle is interpolated using the data from frames before and after the obstruction.

![](Media/steering.gif)

# Usage

Follow these steps to utilize the machine learning models and active learning strategies incorporated in this project:

- [ ] **Prepare Initial Data**
  - Locate the `Src/Feature_Vector_initial.csv` file, which contains 16 features for 58,936 samples across three drivers. This dataset includes annotations for 10% of the data initially.

- [ ] **Train and Evaluate Models**
  - Execute one of the following scripts to train the machine learning model on the annotated dataset and evaluate it on the test dataset (unlabeled data):
    - `Src/XGBoost.py`
    - `Src/Random Forest.py`
    - `Src/MLP.py`
  - These scripts also calculate the threshold for low confidence scores. Samples with the lowest confidence, along with all test samples and their confidence scores, are saved for further processing.

- [ ] **Apply Active Learning Strategy**
  - Run `Src/AL Strategy.py` to implement the Active Learning Strategy. This script selects 3000 samples with the lowest confidence scores from the unlabeled set for relabeling, adhering to our specified budget.

- [ ] **Relabel and Recalculate Metrics**
  - Use `Src/Relabeling.py` to calculate the performance of the model by comparing predicted labels with the actual labels found in `Src/Ground_truth.csv`.
  - This script computes the Confusion Matrix and both Micro and Macro performance metrics.
  - Following the evaluation, the samples selected by the previous script are labeled using ground truth data provided by an Oracle.

The steps outlined above should be repeated until the desired accuracy is achieved.




