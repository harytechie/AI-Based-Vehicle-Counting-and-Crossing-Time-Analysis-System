"""
Vehicle tracking module using SORT algorithm
"""
import numpy as np
from typing import List, Dict, Tuple
from filterpy.kalman import KalmanFilter


class Track:
    """Represents a single tracked vehicle"""
    
    _id_counter = 0
    
    def __init__(self, bbox: List[int], class_id: int, class_name: str):
        """
        Initialize track
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Vehicle class ID
            class_name: Vehicle class name
        """
        Track._id_counter += 1
        self.id = Track._id_counter
        self.class_id = class_id
        self.class_name = class_name
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Initialize Kalman filter for position tracking
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R *= 10.0
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._bbox_to_state(bbox).reshape(4, 1)
        
        # Track history
        self.history = [bbox]
    
    def _bbox_to_state(self, bbox: List[int]) -> np.ndarray:
        """Convert bounding box to state vector [cx, cy, s, r]"""
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bounding box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Calculate area and aspect ratio with safety checks
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        s = width * height  # area (always positive)
        r = width / height  # aspect ratio (always positive)
        
        return np.array([[cx], [cy], [s], [r]])
    
    def _state_to_bbox(self, state: np.ndarray) -> List[int]:
        """Convert state vector to bounding box [x1, y1, x2, y2]"""
        # Extract scalar values from column vector using .item()
        cx = state[0].item() if hasattr(state[0], 'item') else float(state[0])
        cy = state[1].item() if hasattr(state[1], 'item') else float(state[1])
        s = state[2].item() if hasattr(state[2], 'item') else float(state[2])
        r = state[3].item() if hasattr(state[3], 'item') else float(state[3])
        
        # Validate and clamp values to prevent NaN
        s = max(s, 1.0)  # Ensure area is positive
        r = max(r, 0.1)  # Ensure aspect ratio is positive
        
        w = np.sqrt(s * r)
        h = s / max(w, 1.0)
        
        # Check for NaN values
        if np.isnan(cx) or np.isnan(cy) or np.isnan(w) or np.isnan(h):
            # Return a default small bbox if invalid
            return [0, 0, 10, 10]
        
        x1 = int(max(0, cx - w / 2))
        y1 = int(max(0, cy - h / 2))
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return [x1, y1, x2, y2]
    
    def predict(self) -> List[int]:
        """Predict next position"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox(self.kf.x)
    
    def update(self, bbox: List[int]):
        """Update track with new detection"""
        self.kf.update(self._bbox_to_state(bbox))
        self.time_since_update = 0
        self.hits += 1
        self.history.append(bbox)
    
    def get_state(self) -> Dict:
        """Get current track state"""
        bbox = self._state_to_bbox(self.kf.x)
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        return {
            'id': self.id,
            'bbox': bbox,
            'center': center,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'age': self.age,
            'hits': self.hits
        }


class VehicleTracker:
    """Tracks multiple vehicles across frames using SORT algorithm"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker
        
        Args:
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IOU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detections from detector
            
        Returns:
            List of active tracks
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched_indices, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for det_idx, track_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx]['bbox'])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = Track(det['bbox'], det['class_id'], det['class_name'])
            self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return confirmed tracks
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age <= self.min_hits:
                active_tracks.append(track.get_state())
        
        return active_tracks
    
    def _match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks using IOU"""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                predicted_bbox = track._state_to_bbox(track.kf.x)
                iou_matrix[d, t] = self._compute_iou(det['bbox'], predicted_bbox)
        
        # Perform matching using greedy algorithm
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        while True:
            if len(unmatched_detections) == 0 or len(unmatched_tracks) == 0:
                break
            
            # Find maximum IOU
            max_iou = 0
            max_det = -1
            max_track = -1
            
            for d in unmatched_detections:
                for t in unmatched_tracks:
                    if iou_matrix[d, t] > max_iou:
                        max_iou = iou_matrix[d, t]
                        max_det = d
                        max_track = t
            
            # Check if match is good enough
            if max_iou < self.iou_threshold:
                break
            
            # Add match
            matched_indices.append((max_det, max_track))
            unmatched_detections.remove(max_det)
            unmatched_tracks.remove(max_track)
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def get_all_tracks(self) -> List[Dict]:
        """Get all current tracks (including unconfirmed)"""
        return [track.get_state() for track in self.tracks]
    
    def reset(self):
        """Reset tracker"""
        self.tracks = []
        Track._id_counter = 0
