# Workout-Video-Classifier
Workout Video Classifier usinng CNN-LSTM Model 

## Overview
A deep learning classifier that automatically identifies 22 different workout exercises from videos using a CNN-LSTM architecture. Built using a diverse dataset of 652 workout videos, this model is designed to handle real-world challenges including varying video lengths (1-228 seconds), different recording angles, and inconsistent video quality. The project showcases the application of deep learning in fitness technology, processing videos of common exercises like bench press, deadlifts, and biceps curls despite significant variations in how these exercises are recorded.

![Number of Videos per Exercise](/images/Number%20of%20Videos%20per%20Exercise.png)
*Distribution of videos across different exercise types, showing the class imbalance in the dataset.*

## Key Findings
[Summary of results]

## Technologies Used
- Python
- Libraries used (sklearn, tensorflow, etc.)

## How to Run
[Installation and running instructions]

## Dataset
The project utilizes the Workout/Fitness Video dataset from Kaggle, comprising 652 short workout videos across 22 exercise categories. The dataset presents several interesting challenges:

- **Video Distribution**: The dataset has an uneven distribution of exercises, ranging from 62 videos for barbell biceps curl to 7 videos for plank exercises
- **Duration Variability**: Videos vary significantly in length
  - Maximum duration: 228 seconds
  - Minimum duration: 1 second
  - Mean duration: 7.4 seconds
  - Median duration: 5.0 seconds
- **Source Diversity**: Videos are collected from various sources, resulting in:
  - Different recording angles
  - Varying video quality
  - Inconsistent lighting conditions
  - Different environmental settings

## Challenge
The main challenge of this project is to build a robust classifier that can:
1. Handle variable-length input videos
2. Perform well despite the unbalanced class distribution
3. Generalize across different video qualities and recording angles
4. Accurately identify exercises using a CNN-LSTM architecture
