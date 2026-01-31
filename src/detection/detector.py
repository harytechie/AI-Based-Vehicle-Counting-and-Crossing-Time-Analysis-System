"""
Vehicle detection module using YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_config


class VehicleDetector:
    """Detects vehicles in video frames using YOLOv8"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = None):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLO model (if None, uses config)
            confidence_threshold: Confidence threshold for detections (if None, uses config)
        """
        config = get_config()
        
        # Load configuration
        if model_path is None:
            model_name = config.get('model.name', 'yolov8n.pt')
            model_path = f"data/models/{model_name}"
            
            # If model doesn't exist locally, YOLO will download it
            if not Path(model_path).exists():
                model_path = model_name
        
        if confidence_threshold is None:
            confidence_threshold = config.get('model.confidence_threshold', 0.5)
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = config.get('model.iou_threshold', 0.45)
        self.device = config.get('model.device', 'cpu')
        
        # Vehicle class IDs from COCO dataset
        self.vehicle_classes = config.get('detection.vehicle_classes', [2, 3, 5, 7])
        self.class_names = config.get('detection.class_names', {
            2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
        })
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"✓ Loaded YOLOv8 model: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - class_id: vehicle class ID
                - class_name: vehicle class name
        """
        # Run inference
        results = self.model(frame, 
                           conf=self.confidence_threshold,
                           iou=self.iou_threshold,
                           device=self.device,
                           verbose=False)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Filter only vehicle classes
                if class_id not in self.vehicle_classes:
                    continue
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'vehicle'),
                    'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                }
                
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                       show_labels: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_labels: Whether to show class labels
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        # Color map for different vehicle types
        color_map = {
            2: (0, 255, 0),      # car - green
            3: (255, 0, 0),      # motorcycle - blue
            5: (0, 165, 255),    # bus - orange
            7: (0, 255, 255)     # truck - yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Get color for this class
            color = color_map.get(class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = f"{class_name} {confidence:.2f}"
                
                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(output,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Draw text
                cv2.putText(output, label, (x1, y1 - 5),
                           font, font_scale, (0, 0, 0), thickness)
        
        return output
    
    def get_detection_count(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get count of detections by class
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with counts per class
        """
        counts = {name: 0 for name in self.class_names.values()}
        counts['total'] = len(detections)
        
        for det in detections:
            class_name = det['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        
        return counts
