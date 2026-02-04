"""SLAM (Simultaneous Localization and Mapping) module for UAV swarm coordination.

This module implements both SLAM and Visual SLAM (VSLAM) capabilities for UAVs
to build and maintain maps of disaster environments while localizing themselves.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from collections import deque


@dataclass
class Landmark:
    """Represents a landmark in the environment."""
    id: int
    position: np.ndarray  # 3D position (x, y, z)
    observations: int = 0
    descriptor: Optional[np.ndarray] = None  # Visual descriptor for VSLAM
    uncertainty: float = 1.0  # Position uncertainty


@dataclass
class Pose:
    """Represents a UAV pose (position + orientation)."""
    position: np.ndarray  # 3D position (x, y, z)
    orientation: np.ndarray  # Euler angles (roll, pitch, yaw)
    timestamp: float
    uncertainty: np.ndarray  # 6x6 covariance matrix


class EKF_SLAM:
    """Extended Kalman Filter SLAM implementation for UAV navigation."""

    def __init__(self, initial_pose: np.ndarray, process_noise: float = 0.1,
                 measurement_noise: float = 0.05):
        """
        Initialize EKF-SLAM.

        Args:
            initial_pose: Initial UAV pose [x, y, z, roll, pitch, yaw]
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
        """
        self.state_dim = 6  # [x, y, z, roll, pitch, yaw]
        self.state = initial_pose.copy()

        # State covariance matrix
        self.P = np.eye(self.state_dim) * 0.1

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * (process_noise ** 2)

        # Measurement noise covariance
        self.R_landmark = np.eye(3) * (measurement_noise ** 2)

        # Landmarks
        self.landmarks: Dict[int, Landmark] = {}
        self.next_landmark_id = 0

        # SLAM history
        self.pose_history: List[Pose] = []
        self.trajectory: List[np.ndarray] = [initial_pose[:3].copy()]

    def predict(self, control: np.ndarray, dt: float):
        """
        Prediction step: Update pose based on control input (odometry).

        Args:
            control: Control input [vx, vy, vz, angular_velocity_x, angular_velocity_y, angular_velocity_z]
            dt: Time step
        """
        # State transition (simple integration for UAV)
        # Position update
        self.state[0] += control[0] * dt  # x
        self.state[1] += control[1] * dt  # y
        self.state[2] += control[2] * dt  # z

        # Orientation update
        self.state[3] += control[3] * dt  # roll
        self.state[4] += control[4] * dt  # pitch
        self.state[5] += control[5] * dt  # yaw

        # Normalize angles to [-pi, pi]
        self.state[3:6] = np.arctan2(np.sin(self.state[3:6]), np.cos(self.state[3:6]))

        # Jacobian of motion model (simplified)
        F = np.eye(self.state_dim)

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

        # Store trajectory
        self.trajectory.append(self.state[:3].copy())

    def update(self, landmarks_observed: List[Tuple[np.ndarray, Optional[np.ndarray]]]):
        """
        Update step: Correct pose and landmark positions based on observations.

        Args:
            landmarks_observed: List of (position, descriptor) tuples for observed landmarks
        """
        for landmark_pos, descriptor in landmarks_observed:
            # Data association: find matching landmark or create new one
            landmark_id = self._associate_landmark(landmark_pos, descriptor)

            if landmark_id is None:
                # New landmark
                self._add_landmark(landmark_pos, descriptor)
            else:
                # Update existing landmark
                self._update_landmark(landmark_id, landmark_pos)

    def _associate_landmark(self, observed_pos: np.ndarray,
                           descriptor: Optional[np.ndarray] = None,
                           threshold: float = 5.0) -> Optional[int]:
        """
        Associate observed landmark with existing landmarks.

        Returns:
            Landmark ID if match found, None otherwise
        """
        if len(self.landmarks) == 0:
            return None

        # Transform observation to global frame
        global_pos = self._observation_to_global(observed_pos)

        # Find closest landmark
        min_distance = float('inf')
        best_match = None

        for lid, landmark in self.landmarks.items():
            distance = np.linalg.norm(global_pos - landmark.position)

            # Use descriptor matching if available (VSLAM)
            if descriptor is not None and landmark.descriptor is not None:
                descriptor_similarity = self._descriptor_similarity(descriptor, landmark.descriptor)
                # Combine distance and descriptor similarity
                score = distance * (1.0 - descriptor_similarity * 0.5)
            else:
                score = distance

            if score < min_distance:
                min_distance = score
                best_match = lid

        if min_distance < threshold:
            return best_match

        return None

    def _descriptor_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Compute descriptor similarity (normalized cross-correlation)."""
        if desc1 is None or desc2 is None:
            return 0.0

        # Normalize descriptors
        desc1_norm = desc1 / (np.linalg.norm(desc1) + 1e-8)
        desc2_norm = desc2 / (np.linalg.norm(desc2) + 1e-8)

        # Cosine similarity
        similarity = np.dot(desc1_norm, desc2_norm)
        return max(0.0, similarity)

    def _observation_to_global(self, observed_pos: np.ndarray) -> np.ndarray:
        """Transform observation from UAV frame to global frame."""
        # Simple transformation (can be extended with full rotation matrix)
        yaw = self.state[5]
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        global_pos = self.state[:3] + rotation @ observed_pos
        return global_pos

    def _add_landmark(self, observed_pos: np.ndarray, descriptor: Optional[np.ndarray] = None):
        """Add a new landmark to the map."""
        global_pos = self._observation_to_global(observed_pos)

        landmark = Landmark(
            id=self.next_landmark_id,
            position=global_pos.copy(),
            observations=1,
            descriptor=descriptor.copy() if descriptor is not None else None,
            uncertainty=1.0
        )

        self.landmarks[self.next_landmark_id] = landmark
        self.next_landmark_id += 1

    def _update_landmark(self, landmark_id: int, observed_pos: np.ndarray):
        """Update existing landmark position using EKF update."""
        landmark = self.landmarks[landmark_id]

        # Transform observation to global frame
        observed_global = self._observation_to_global(observed_pos)

        # Innovation (measurement residual)
        innovation = observed_global - landmark.position

        # Measurement Jacobian (simplified)
        H = np.eye(3)

        # Innovation covariance
        S = H @ np.eye(3) * landmark.uncertainty @ H.T + self.R_landmark

        # Kalman gain
        K = np.eye(3) * landmark.uncertainty @ H.T @ np.linalg.inv(S)

        # Update landmark position
        landmark.position += K @ innovation

        # Update landmark uncertainty
        landmark.uncertainty = (1 - np.trace(K @ H) / 3) * landmark.uncertainty

        # Increment observation count
        landmark.observations += 1

    def get_map(self) -> Dict[str, Any]:
        """
        Get current map representation.

        Returns:
            Dictionary containing map data
        """
        return {
            'pose': self.state.copy(),
            'landmarks': {
                lid: {
                    'position': lm.position.copy(),
                    'observations': lm.observations,
                    'uncertainty': lm.uncertainty
                }
                for lid, lm in self.landmarks.items()
            },
            'trajectory': np.array(self.trajectory),
            'covariance': self.P.copy()
        }

    def get_pose(self) -> np.ndarray:
        """Get current pose estimate."""
        return self.state.copy()

    def get_pose_uncertainty(self) -> np.ndarray:
        """Get current pose uncertainty (covariance)."""
        return self.P.copy()


class VisualSLAM:
    """Visual SLAM implementation using feature detection and matching."""

    def __init__(self, initial_pose: np.ndarray, camera_matrix: Optional[np.ndarray] = None):
        """
        Initialize Visual SLAM.

        Args:
            initial_pose: Initial UAV pose [x, y, z, roll, pitch, yaw]
            camera_matrix: Camera intrinsic matrix (3x3)
        """
        # Use EKF-SLAM as backend
        self.slam = EKF_SLAM(initial_pose)

        # Camera parameters
        if camera_matrix is None:
            # Default camera matrix (can be calibrated)
            focal_length = 500.0
            cx, cy = 320.0, 240.0  # Image center
            self.camera_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ])
        else:
            self.camera_matrix = camera_matrix

        # Feature detector (ORB - fast and rotation invariant)
        self.feature_detector = cv2.ORB_create(nfeatures=500)

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_image = None

        # Map points (3D landmarks from visual features)
        self.map_points: List[np.ndarray] = []
        self.map_descriptors: List[np.ndarray] = []

    def process_frame(self, image: np.ndarray, control: np.ndarray, dt: float) -> Dict[str, Any]:
        """
        Process a new camera frame.

        Args:
            image: Grayscale or RGB image from camera
            control: Control input [vx, vy, vz, wx, wy, wz]
            dt: Time step

        Returns:
            Dictionary with tracking results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect features
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

        # Prediction step
        self.slam.predict(control, dt)

        results = {
            'pose': self.slam.get_pose(),
            'num_features': len(keypoints),
            'num_landmarks': len(self.slam.landmarks),
            'tracking_quality': 0.0
        }

        if self.prev_descriptors is not None and descriptors is not None:
            # Match features with previous frame
            matches = self.matcher.match(self.prev_descriptors, descriptors)

            if len(matches) > 10:  # Minimum matches for reliable tracking
                # Extract matched points
                prev_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
                curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])

                # Estimate motion (Essential matrix for calibrated camera)
                E, mask = cv2.findEssentialMat(prev_pts, curr_pts, self.camera_matrix,
                                               method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None:
                    # Recover pose
                    _, R, t, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.camera_matrix, mask=mask)

                    # Triangulate 3D points
                    landmarks = self._triangulate_points(prev_pts[mask.ravel() == 1],
                                                        curr_pts[mask.ravel() == 1], R, t)

                    # Update SLAM with observed landmarks
                    landmark_observations = []
                    for i, landmark_3d in enumerate(landmarks):
                        # Extract descriptor for this landmark
                        match_idx = np.where(mask.ravel() == 1)[0][i]
                        descriptor = descriptors[matches[match_idx].trainIdx]
                        landmark_observations.append((landmark_3d, descriptor))

                    self.slam.update(landmark_observations)

                    # Update tracking quality
                    results['tracking_quality'] = len(landmarks) / len(keypoints)
                    results['num_matches'] = len(matches)

        # Store current frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_image = gray

        return results

    def _triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                           R: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
        """
        Triangulate 3D points from matched 2D points.

        Args:
            pts1: Points from first image (Nx2)
            pts2: Points from second image (Nx2)
            R: Rotation matrix from camera 1 to camera 2
            t: Translation vector from camera 1 to camera 2

        Returns:
            List of 3D points in camera frame
        """
        # Projection matrices
        P1 = self.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.camera_matrix @ np.hstack([R, t])

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3]

        # Convert to list of 3D points
        landmarks = []
        for i in range(points_3d.shape[1]):
            point = points_3d[:, i]
            # Filter out points that are too far or behind camera
            if 0.1 < point[2] < 100.0:  # Reasonable depth range
                landmarks.append(point)

        return landmarks

    def get_map(self) -> Dict[str, Any]:
        """Get current VSLAM map."""
        slam_map = self.slam.get_map()
        slam_map['num_features'] = len(self.map_points)
        return slam_map

    def get_pose(self) -> np.ndarray:
        """Get current pose estimate."""
        return self.slam.get_pose()


class CollaborativeSLAM:
    """Multi-UAV collaborative SLAM for shared map building."""

    def __init__(self, num_uavs: int, initial_poses: List[np.ndarray]):
        """
        Initialize collaborative SLAM.

        Args:
            num_uavs: Number of UAVs in the swarm
            initial_poses: List of initial poses for each UAV
        """
        self.num_uavs = num_uavs

        # Individual SLAM instances for each UAV
        self.uav_slams: List[EKF_SLAM] = [
            EKF_SLAM(pose) for pose in initial_poses
        ]

        # Shared global map
        self.global_landmarks: Dict[int, Landmark] = {}
        self.next_global_landmark_id = 0

        # Map merging parameters
        self.merge_threshold = 2.0  # Distance threshold for merging landmarks

    def update_uav(self, uav_id: int, control: np.ndarray, dt: float,
                   observations: List[Tuple[np.ndarray, Optional[np.ndarray]]]):
        """
        Update SLAM for a specific UAV.

        Args:
            uav_id: UAV identifier
            control: Control input
            dt: Time step
            observations: List of landmark observations
        """
        slam = self.uav_slams[uav_id]

        # Predict and update
        slam.predict(control, dt)
        slam.update(observations)

        # Merge with global map
        self._merge_to_global_map(uav_id)

    def _merge_to_global_map(self, uav_id: int):
        """Merge UAV's local map with global map."""
        slam = self.uav_slams[uav_id]

        for local_id, landmark in slam.landmarks.items():
            # Check if landmark already exists in global map
            matched = False

            for global_id, global_landmark in self.global_landmarks.items():
                distance = np.linalg.norm(landmark.position - global_landmark.position)

                # Use descriptor matching if available
                if (landmark.descriptor is not None and
                    global_landmark.descriptor is not None):
                    similarity = self._descriptor_similarity(
                        landmark.descriptor, global_landmark.descriptor
                    )
                    score = distance * (1.0 - similarity * 0.5)
                else:
                    score = distance

                if score < self.merge_threshold:
                    # Merge landmarks (weighted average based on observations)
                    total_obs = landmark.observations + global_landmark.observations
                    global_landmark.position = (
                        landmark.position * landmark.observations +
                        global_landmark.position * global_landmark.observations
                    ) / total_obs
                    global_landmark.observations = total_obs
                    matched = True
                    break

            if not matched:
                # Add new landmark to global map
                global_landmark = Landmark(
                    id=self.next_global_landmark_id,
                    position=landmark.position.copy(),
                    observations=landmark.observations,
                    descriptor=landmark.descriptor.copy() if landmark.descriptor is not None else None,
                    uncertainty=landmark.uncertainty
                )
                self.global_landmarks[self.next_global_landmark_id] = global_landmark
                self.next_global_landmark_id += 1

    def _descriptor_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Compute descriptor similarity."""
        desc1_norm = desc1 / (np.linalg.norm(desc1) + 1e-8)
        desc2_norm = desc2 / (np.linalg.norm(desc2) + 1e-8)
        return max(0.0, np.dot(desc1_norm, desc2_norm))

    def get_global_map(self) -> Dict[str, Any]:
        """Get the merged global map."""
        return {
            'num_landmarks': len(self.global_landmarks),
            'landmarks': {
                lid: {
                    'position': lm.position.copy(),
                    'observations': lm.observations,
                    'uncertainty': lm.uncertainty
                }
                for lid, lm in self.global_landmarks.items()
            },
            'uav_poses': [slam.get_pose() for slam in self.uav_slams],
            'uav_trajectories': [slam.trajectory for slam in self.uav_slams]
        }

    def get_uav_pose(self, uav_id: int) -> np.ndarray:
        """Get pose estimate for specific UAV."""
        return self.uav_slams[uav_id].get_pose()


def create_synthetic_landmarks(environment_size: Tuple[float, float, float],
                               num_landmarks: int, seed: int = 42) -> List[np.ndarray]:
    """
    Create synthetic landmarks for testing SLAM.

    Args:
        environment_size: (width, height, max_altitude)
        num_landmarks: Number of landmarks to generate
        seed: Random seed

    Returns:
        List of 3D landmark positions
    """
    np.random.seed(seed)

    landmarks = []
    for _ in range(num_landmarks):
        x = np.random.uniform(0, environment_size[0])
        y = np.random.uniform(0, environment_size[1])
        z = np.random.uniform(0, environment_size[2] * 0.5)  # Landmarks on ground/structures
        landmarks.append(np.array([x, y, z]))

    return landmarks
