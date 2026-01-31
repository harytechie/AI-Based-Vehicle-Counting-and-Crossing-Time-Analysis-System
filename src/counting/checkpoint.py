"""
Checkpoint management and crossing detection
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import get_config


class Checkpoint:
    """Manages checkpoint line/region for vehicle crossing detection"""
    
    def __init__(self, checkpoint_type: str = "line"):
        """
        Initialize checkpoint
        
        Args:
            checkpoint_type: Type of checkpoint ('line' or 'polygon')
        """
        config = get_config()
        self.type = checkpoint_type
        self.line_start: Optional[Tuple[int, int]] = None
        self.line_end: Optional[Tuple[int, int]] = None
        self.polygon_points: List[Tuple[int, int]] = []
        self.crossing_buffer = config.get('checkpoint.crossing_buffer', 5)
        
        # Load default checkpoint if configured
        if checkpoint_type == "line":
            default_line = config.get('checkpoint.default_line', {})
            if default_line:
                self.line_start = tuple(default_line.get('start', [100, 300]))
                self.line_end = tuple(default_line.get('end', [500, 300]))
    
    def set_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Set checkpoint line"""
        self.type = "line"
        self.line_start = start
        self.line_end = end
    
    def set_polygon(self, points: List[Tuple[int, int]]):
        """Set checkpoint polygon region"""
        self.type = "polygon"
        self.polygon_points = points
    
    def check_crossing(self, prev_position: Tuple[int, int], 
                      curr_position: Tuple[int, int]) -> Optional[str]:
        """
        Check if vehicle crossed checkpoint
        
        Args:
            prev_position: Previous position (x, y)
            curr_position: Current position (x, y)
            
        Returns:
            'up' if crossed upward, 'down' if crossed downward, None if no crossing
        """
        if self.type == "line" and self.line_start and self.line_end:
            return self._check_line_crossing(prev_position, curr_position)
        elif self.type == "polygon" and self.polygon_points:
            return self._check_polygon_crossing(prev_position, curr_position)
        return None
    
    def _check_line_crossing(self, prev_pos: Tuple[int, int], 
                            curr_pos: Tuple[int, int]) -> Optional[str]:
        """Check if trajectory crosses the checkpoint line"""
        if not self.line_start or not self.line_end:
            return None
        
        # Check if line segment (prev_pos, curr_pos) intersects checkpoint line
        x1, y1 = prev_pos
        x2, y2 = curr_pos
        x3, y3 = self.line_start
        x4, y4 = self.line_end
        
        # Compute cross products
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Check if segments intersect
        if ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and \
           ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4)):
            # Determine direction
            if y2 < y1:  # Moving upward
                return 'up'
            else:  # Moving downward
                return 'down'
        
        return None
    
    def _check_polygon_crossing(self, prev_pos: Tuple[int, int], 
                               curr_pos: Tuple[int, int]) -> Optional[str]:
        """Check if vehicle entered/exited polygon region"""
        if not self.polygon_points:
            return None
        
        # Check if points are inside polygon
        prev_inside = self._point_in_polygon(prev_pos)
        curr_inside = self._point_in_polygon(curr_pos)
        
        # Crossing detected if state changed
        if not prev_inside and curr_inside:
            return 'enter'
        elif prev_inside and not curr_inside:
            return 'exit'
        
        return None
    
    def _point_in_polygon(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(self.polygon_points)
        inside = False
        
        p1x, p1y = self.polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255),
            thickness: int = 3) -> np.ndarray:
        """
        Draw checkpoint on frame
        
        Args:
            frame: Input frame
            color: Line color (B, G, R)
            thickness: Line thickness
            
        Returns:
            Frame with checkpoint drawn
        """
        output = frame.copy()
        
        if self.type == "line" and self.line_start and self.line_end:
            cv2.line(output, self.line_start, self.line_end, color, thickness)
            
            # Draw endpoints
            cv2.circle(output, self.line_start, 5, color, -1)
            cv2.circle(output, self.line_end, 5, color, -1)
            
            # Add label
            mid_point = (
                (self.line_start[0] + self.line_end[0]) // 2,
                (self.line_start[1] + self.line_end[1]) // 2
            )
            cv2.putText(output, "CHECKPOINT", 
                       (mid_point[0] - 50, mid_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        elif self.type == "polygon" and len(self.polygon_points) > 2:
            points = np.array(self.polygon_points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(output, [points], True, color, thickness)
        
        return output
    
    def is_configured(self) -> bool:
        """Check if checkpoint is properly configured"""
        if self.type == "line":
            return self.line_start is not None and self.line_end is not None
        elif self.type == "polygon":
            return len(self.polygon_points) >= 3
        return False
