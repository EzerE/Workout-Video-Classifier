# Workout-Video-Classifier
Workout Video Classifier using CNN-LSTM Model 

## Overview
A deep learning classifier that automatically identifies 22 different workout exercises from videos using a CNN-LSTM architecture. Built using a diverse dataset of 652 workout videos, this model is designed to handle real-world challenges including varying video lengths (1-228 seconds), different recording angles, and inconsistent video quality. The project showcases the application of deep learning in fitness technology, processing videos of common exercises like bench press, deadlifts, and biceps curls despite significant variations in how these exercises are recorded.

![Number of Videos per Exercise](/images/Number%20of%20Videos%20per%20Exercise.png)
*Distribution of videos across different exercise types, showing the class imbalance in the dataset.*
![overall_duration_distribution](/images/overall_duration_distribution.png)
*Video duration distribution showing most samples are between 5-10 seconds, with notable outliers reaching up to 228 seconds. The median duration of 5 seconds indicates that most exercises were captured in short clips.*

## Results
### Key Findings
- **High Overall Accuracy**: Achieved 94.28% (±2.18%) accuracy across 5-fold cross-validation
- **Strong Class Performance**: The model excelled at distinguishing most exercises, with perfect recognition for:
  - Compound movements (deadlift, squat)
  - Isolation exercises (leg extension, leg raises)
  - Bodyweight exercises (push-up, plank)
- **Common Confusions**: The model occasionally confused:
  - Barbell biceps curl with tricep pushdown
  - Bench press variants (regular, incline, and decline)
  - Similar movement patterns within muscle groups
- **Robust Learning**: Demonstrated consistent performance across all folds (accuracy range: 91-97%)
- **Challenging Cases**: Lower accuracy on decline bench press, likely due to limited training samples (only 4 samples per fold)

![training curves](/images/training_curves.png)
*Model training and validation curves showing steady learning progression. The close alignment between training and validation curves indicates good generalization without overfitting. Model achieves stable performance around 80 epochs, with accuracy consistently above 90%.*

![Confusion Matrix](/images/confusion_matrix.png)
*Confusion matrix showing model predictions vs actual exercise classes. Darker blue indicates more samples correctly classified.*

## Methodology & Implementation

### Data Processing Pipeline
![Processing Pipeline](/images/processing%20new.png)
*Data processing pipeline showing feature engineering steps: from raw video input through frame extraction, pose detection, and feature combination. The pipeline incorporates normalized poses, velocity features, and weighted confidence features.*

### System Architecture
![System Architecture](/images/model-diagram.png)
*System architecture overview showing four main components: data preparation, model architecture, training process, and evaluation pipeline.*

### Model Components
- **Convolutional Layer**
  - 1D convolution with 64 filters
  - BatchNormalization and MaxPooling
  - Dropout for regularization
- **LSTM Layer**
  - 128 LSTM units for temporal feature extraction
  - 64-unit second layer for sequence processing
  - BatchNormalization and Dropout
- **Dense Layer**
  - 64 units with BatchNormalization
  - Softmax activation for 22-class classification

### Training Configuration
- Batch size: 16
- 5-fold cross-validation
- Class weights to handle imbalanced data
- Custom callbacks for monitoring and optimization

### Technologies & Libraries
- **Core Processing**
  - TensorFlow: Deep learning framework
  - MediaPipe: Pose estimation and landmark extraction
  - OpenCV: Video processing and frame extraction

- **Data & Analysis**
  - NumPy: Numerical computations
  - Pandas: Data manipulation
  - Scikit-learn: Cross-validation and metrics
  - Collections: Data structures

- **Visualization**
  - Matplotlib & Seaborn: Performance plots
  - IPython: Development interface

## Repository Structure and Implementation

### Code Organization
The implementation is divided into three main components:

1. **Data Analysis** (`src/data_analysis/`)
   - Dataset statistics computation
   - Data distribution visualization
   - Quality analysis tools

2. **Preprocessing** (`src/preprocessing/`)
   - Video frame extraction
   - Pose detection using MediaPipe
   - Feature engineering pipeline
   - Data augmentation

3. **Model Training** (`src/training/`)
   - CNN-LSTM model implementation
   - Training configuration
   - Cross-validation
   - Performance evaluation

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/workout-video-classifier
   cd workout-video-classifier

## How to Run
### Prerequisites
- Python 3.8+
- Kaggle account (for dataset download)
- Required libraries: TensorFlow, MediaPipe, OpenCV, etc.

### Dataset Setup
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
2. Extract the downloaded zip file to the `data` directory:
   ```bash
   unzip workout-fitness-video.zip -d data/

## Dataset Source
The dataset used in this project is the ["Workout/Fitness Video"](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video) dataset from Kaggle, created by Hasyim Abdillah. It contains 652 workout videos across 22 exercise categories, making it suitable for fitness movement classification tasks.

The dataset presents several interesting challenges:

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
