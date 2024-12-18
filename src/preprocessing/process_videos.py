import os
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict

# Define landmark groups and exercise weights
LANDMARK_GROUPS = {
    'arms': [11,12,13,14,15,16],  # shoulder to wrist landmarks
    'legs': [23,24,25,26,27,28],  # hip to ankle landmarks
    'core': [23,24,11,12],  # hips and shoulders
    'upper_body': [11,12,13,14,15,16,0,1,2,3,4,5,6,7,8,9,10]  # above hips
}

EXERCISE_WEIGHTS = {
    'bench press': {'arms': 0.5, 'core': 0.3, 'legs': 0.2},
    'barbell biceps curl': {'arms': 0.7, 'core': 0.2, 'legs': 0.1},
    'chest fly machine': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
    'deadlift': {'legs': 0.4, 'core': 0.4, 'arms': 0.2},
    'decline bench press': {'arms': 0.5, 'core': 0.3, 'legs': 0.2},
    'hammer curl': {'arms': 0.7, 'core': 0.2, 'legs': 0.1},
    'hip thrust': {'legs': 0.5, 'core': 0.4, 'arms': 0.1},
    'incline bench press': {'arms': 0.5, 'core': 0.3, 'legs': 0.2},
    'lat pulldown': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
    'lateral raise': {'arms': 0.6, 'core': 0.3, 'legs': 0.1},
    'leg extension': {'legs': 0.7, 'core': 0.2, 'arms': 0.1},
    'leg raises': {'core': 0.6, 'legs': 0.3, 'arms': 0.1},
    'plank': {'core': 0.7, 'arms': 0.2, 'legs': 0.1},
    'pull Up': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
    'push-up': {'arms': 0.5, 'core': 0.4, 'legs': 0.1},
    'romanian deadlift': {'legs': 0.5, 'core': 0.3, 'arms': 0.2},
    'russian twist': {'core': 0.6, 'arms': 0.2, 'legs': 0.2},
    'shoulder press': {'arms': 0.6, 'core': 0.3, 'legs': 0.1},
    'squat': {'legs': 0.6, 'core': 0.3, 'arms': 0.1},
    't bar row': {'arms': 0.4, 'core': 0.4, 'legs': 0.2},
    'tricep dips': {'arms': 0.6, 'core': 0.3, 'legs': 0.1},
    'tricep Pushdown': {'arms': 0.7, 'core': 0.2, 'legs': 0.1}
}

class WorkoutDataPreparator:
    def __init__(self, base_folder, target_classes=None, max_frames=45):
        self.base_folder = base_folder
        self.target_classes = target_classes
        self.max_frames = max_frames
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2, #model 2 insted of 1
            min_detection_confidence=0.5
        )

    def calculate_weighted_confidence(self, landmarks, exercise_type):
        weights = EXERCISE_WEIGHTS[exercise_type]
        weighted_scores = np.zeros(33)  # One score per landmark
        
        # Calculate weighted confidence for each landmark
        for group, weight in weights.items():
            group_landmarks = LANDMARK_GROUPS[group]
            for lm_idx in group_landmarks:
                weighted_scores[lm_idx] = landmarks[lm_idx].visibility * weight
                
        return weighted_scores
    
    def normalize_poses(self, poses):
        normalized_poses = []
        for pose in poses:
            landmarks = pose.reshape(-1, 4)
            left_hip = landmarks[23][:3]
            right_hip = landmarks[24][:3]
            hip_center = (left_hip + right_hip) / 2
            normalized_landmarks = landmarks.copy()
            normalized_landmarks[:, :3] -= hip_center
            normalized_poses.append(normalized_landmarks.flatten())
        return np.array(normalized_poses)

    def calculate_velocity(self, poses):
        velocity = np.zeros_like(poses)
        velocity[1:] = poses[1:] - poses[:-1]
        return velocity

    def extract_poses_from_video(self, video_path, exercise_type):
        poses = []
        confidence_scores = []
        weighted_confidences = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Original confidence calculation
                    confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
                    
                    # Calculate weighted confidence scores
                    weighted_conf = self.calculate_weighted_confidence(
                        results.pose_landmarks.landmark, 
                        exercise_type
                    )
                    
                    if confidence > 0.4:
                        landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                                           for lm in results.pose_landmarks.landmark])
                        poses.append(landmarks.flatten())
                        confidence_scores.append(confidence)
                        weighted_confidences.append(weighted_conf)
                    else:
                        poses.append(np.zeros(33 * 4))
                        confidence_scores.append(0.0)
                        weighted_confidences.append(np.zeros(33))
                else:
                    poses.append(np.zeros(33 * 4))
                    confidence_scores.append(0.0)
                    weighted_confidences.append(np.zeros(33))
            
            cap.release()
            
            if len(poses) < self.max_frames:
                last_pose = poses[-1] if poses else np.zeros(33 * 4)
                last_weighted_conf = weighted_confidences[-1] if weighted_confidences else np.zeros(33)
                poses.extend([last_pose] * (self.max_frames - len(poses)))
                confidence_scores.extend([0.0] * (self.max_frames - len(confidence_scores)))
                weighted_confidences.extend([last_weighted_conf] * (self.max_frames - len(weighted_confidences)))
            
            poses = np.array(poses)
            normalized_poses = self.normalize_poses(poses)
            velocity_features = self.calculate_velocity(normalized_poses)
            weighted_conf_features = np.array(weighted_confidences)
            
            # Concatenate all features
            final_poses = np.concatenate([normalized_poses, velocity_features, weighted_conf_features], axis=1)
            
            return final_poses, np.array(confidence_scores)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return np.zeros((self.max_frames, 33 * 4 * 2 + 33)), np.zeros(self.max_frames)  # Updated shape

    def mirror_augment(self, poses):
        mirrored = poses.copy()
        # Split into components (now including weighted confidence)
        orig_poses = mirrored[:, :33*4]
        velocity = mirrored[:, 33*4:33*8]
        weighted_conf = mirrored[:, 33*8:]  # Last 33 features are weighted confidence
        
        landmarks = orig_poses.reshape(-1, 33, 4)
        landmarks[:, :, 0] = 1 - landmarks[:, :, 0]  # Mirror x coordinates
        
        pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16),
                (23,24), (25,26), (27,28), (29,30), (31,32)]
        
        for pair in pairs:
            landmarks[:, [pair[0], pair[1]]] = landmarks[:, [pair[1], pair[0]]]
            # Mirror weighted confidences for paired landmarks
            weighted_conf[:, [pair[0], pair[1]]] = weighted_conf[:, [pair[1], pair[0]]]
        
        vel_landmarks = velocity.reshape(-1, 33, 4)
        vel_landmarks[:, :, 0] *= -1  # Mirror x velocities
        for pair in pairs:
            vel_landmarks[:, [pair[0], pair[1]]] = vel_landmarks[:, [pair[1], pair[0]]]
            
        mirrored_poses = landmarks.reshape(poses.shape[0], -1)
        mirrored_velocity = vel_landmarks.reshape(poses.shape[0], -1)
        
        # Concatenate all components back together
        return np.concatenate([mirrored_poses, mirrored_velocity, weighted_conf], axis=1)

    def prepare_and_save_dataset(self, save_dir):
        X, y = [], []
        confidences = []
        class_mapping = {}
        skipped_videos = defaultdict(int)
        
        print("Preparing dataset...")
        os.makedirs(save_dir, exist_ok=True)
        
        for root, _, files in os.walk(self.base_folder):
            folder_name = os.path.basename(root)
            
            if self.target_classes and folder_name not in self.target_classes:
                continue
                
            if folder_name not in class_mapping:
                class_mapping[folder_name] = len(class_mapping)
            
            for file in files:
                if file.lower().endswith(('.mp4', '.mov', '.avi')):
                    video_path = os.path.join(root, file)
                    try:
                        pose_sequence, conf_scores = self.extract_poses_from_video(video_path, folder_name)
                        
                        if np.mean(conf_scores) > 0.2:
                            X.append(pose_sequence)
                            y.append(class_mapping[folder_name])
                            confidences.append(np.mean(conf_scores))
                            
                            mirrored_sequence = self.mirror_augment(pose_sequence)
                            X.append(mirrored_sequence)
                            y.append(class_mapping[folder_name])
                            confidences.append(np.mean(conf_scores))
                        else:
                            skipped_videos[folder_name] += 1
                            
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")
                        skipped_videos[folder_name] += 1
        
        X = np.array(X)
        y = np.array(y)
        confidences = np.array(confidences)
        
        print("\nDataset Statistics:")
        for class_name, class_idx in class_mapping.items():
            total = sum(1 for label in y if label == class_idx)
            print(f"{class_name}: {total} samples")
        
        print("\nFinal X shape:", X.shape)  # Added to verify shape
        
        np.save(os.path.join(save_dir, 'X.npy'), X)
        np.save(os.path.join(save_dir, 'y.npy'), y)
        np.save(os.path.join(save_dir, 'confidences.npy'), confidences)
        np.save(os.path.join(save_dir, 'class_mapping.npy'), class_mapping)
        
        print(f"\nData saved to {save_dir}")
        return X, y, class_mapping, confidences


if __name__ == "__main__":
    base_folder = r"C:\Users\user\Documents\Workout"
    target_classes = ['bench press', 'barbell biceps curl', 'chest fly machine', 'deadlift',
                     'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press',
                     'lat pulldown', 'lateral raise', 'leg extension', 'leg raises', 'plank',
                     'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 'shoulder press',
                     'squat', 't bar row', 'tricep dips', 'tricep Pushdown']
    save_dir = r"C:\Users\user\Documents\workout_processed_data_landmark"
    
    preparator = WorkoutDataPreparator(
        base_folder=base_folder,
        target_classes=target_classes,
        max_frames=45
    )
    
    X, y, class_mapping, confidences = preparator.prepare_and_save_dataset(save_dir)
