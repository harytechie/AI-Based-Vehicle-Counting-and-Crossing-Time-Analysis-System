"""
Vehicle counting logic
"""
from typing import Dict, List, Optional, Tuple
from .checkpoint import Checkpoint


class VehicleCounter:
    """Counts vehicles crossing checkpoint"""
    
    def __init__(self, checkpoint: Checkpoint):
        """
        Initialize counter
        
        Args:
            checkpoint: Checkpoint instance
        """
        self.checkpoint = checkpoint
        self.crossed_vehicles: Dict[int, Dict] = {}  # track_id -> crossing info
        self.vehicle_positions: Dict[int, Tuple[int, int]] = {}  # track_id -> last position
        
        # Statistics
        self.total_count = 0
        self.direction_counts = {'up': 0, 'down': 0, 'enter': 0, 'exit': 0}
    
    def update(self, tracks: List[Dict]) -> List[Dict]:
        """
        Update counter with current tracks
        
        Args:
            tracks: List of active tracks
            
        Returns:
            List of new crossing events
        """
        new_crossings = []
        
        for track in tracks:
            track_id = track['id']
            curr_position = track['center']
            
            # Get previous position
            prev_position = self.vehicle_positions.get(track_id)
            
            if prev_position is not None:
                # Check for crossing
                direction = self.checkpoint.check_crossing(prev_position, curr_position)
                
                if direction is not None:
                    # Check if not already counted
                    if track_id not in self.crossed_vehicles:
                        crossing_event = {
                            'track_id': track_id,
                            'class_name': track['class_name'],
                            'direction': direction,
                            'position': curr_position
                        }
                        
                        self.crossed_vehicles[track_id] = crossing_event
                        self.total_count += 1
                        self.direction_counts[direction] = self.direction_counts.get(direction, 0) + 1
                        
                        new_crossings.append(crossing_event)
            
            # Update position
            self.vehicle_positions[track_id] = curr_position
        
        return new_crossings
    
    def get_total_count(self) -> int:
        """Get total vehicle count"""
        return self.total_count
    
    def get_direction_counts(self) -> Dict[str, int]:
        """Get counts by direction"""
        return self.direction_counts.copy()
    
    def get_crossed_vehicles(self) -> Dict[int, Dict]:
        """Get all crossed vehicles"""
        return self.crossed_vehicles.copy()
    
    def reset(self):
        """Reset counter"""
        self.crossed_vehicles.clear()
        self.vehicle_positions.clear()
        self.total_count = 0
        self.direction_counts = {'up': 0, 'down': 0, 'enter': 0, 'exit': 0}
