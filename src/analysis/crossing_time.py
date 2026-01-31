"""
Crossing time analysis module
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json


class CrossingTimeAnalyzer:
    """Analyzes vehicle crossing times"""
    
    def __init__(self, fps: int = 30):
        """
        Initialize analyzer
        
        Args:
            fps: Video frames per second for time calculation
        """
        self.fps = fps
        self.entry_times: Dict[int, float] = {}  # track_id -> entry timestamp
        self.exit_times: Dict[int, float] = {}   # track_id -> exit timestamp
        self.crossing_data: List[Dict] = []      # Complete crossing records
    
    def record_crossing(self, track_id: int, frame_number: int, 
                       class_name: str, direction: str):
        """
        Record crossing event
        
        Args:
            track_id: Vehicle track ID
            frame_number: Frame number when crossing occurred
            class_name: Vehicle class name
            direction: Crossing direction
        """
        timestamp = frame_number / self.fps
        
        # For line crossing (up/down), record as both entry and exit
        if direction in ['up', 'down']:
            if track_id not in self.entry_times:
                self.entry_times[track_id] = timestamp
            self.exit_times[track_id] = timestamp
            
            # Calculate crossing time
            if track_id in self.entry_times:
                crossing_time = timestamp - self.entry_times[track_id]
                
                self.crossing_data.append({
                    'track_id': track_id,
                    'class_name': class_name,
                    'entry_time': self.entry_times[track_id],
                    'exit_time': timestamp,
                    'crossing_time': crossing_time,
                    'direction': direction,
                    'entry_frame': int(self.entry_times[track_id] * self.fps),
                    'exit_frame': frame_number
                })
        
        # For polygon crossing (enter/exit)
        elif direction == 'enter':
            self.entry_times[track_id] = timestamp
        
        elif direction == 'exit':
            if track_id in self.entry_times:
                entry_time = self.entry_times[track_id]
                crossing_time = timestamp - entry_time
                
                self.crossing_data.append({
                    'track_id': track_id,
                    'class_name': class_name,
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'crossing_time': crossing_time,
                    'direction': 'enter->exit',
                    'entry_frame': int(entry_time * self.fps),
                    'exit_frame': frame_number
                })
                
                self.exit_times[track_id] = timestamp
    
    def get_crossing_time(self, track_id: int) -> Optional[float]:
        """
        Get crossing time for a specific vehicle
        
        Args:
            track_id: Vehicle track ID
            
        Returns:
            Crossing time in seconds or None if not available
        """
        if track_id in self.entry_times and track_id in self.exit_times:
            return self.exit_times[track_id] - self.entry_times[track_id]
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get crossing time statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.crossing_data:
            return {
                'total_vehicles': 0,
                'average_crossing_time': 0,
                'min_crossing_time': 0,
                'max_crossing_time': 0,
                'median_crossing_time': 0
            }
        
        crossing_times = [record['crossing_time'] for record in self.crossing_data]
        
        return {
            'total_vehicles': len(self.crossing_data),
            'average_crossing_time': sum(crossing_times) / len(crossing_times),
            'min_crossing_time': min(crossing_times),
            'max_crossing_time': max(crossing_times),
            'median_crossing_time': sorted(crossing_times)[len(crossing_times) // 2],
            'std_deviation': self._calculate_std(crossing_times)
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_all_crossing_data(self) -> List[Dict]:
        """Get all crossing records"""
        return self.crossing_data.copy()
    
    def export_to_csv(self, output_path: str):
        """
        Export crossing data to CSV
        
        Args:
            output_path: Output CSV file path
        """
        if not self.crossing_data:
            print("No crossing data to export")
            return
        
        df = pd.DataFrame(self.crossing_data)
        df.to_csv(output_path, index=False)
        print(f"✓ Exported crossing data to {output_path}")
    
    def export_to_json(self, output_path: str):
        """
        Export crossing data to JSON
        
        Args:
            output_path: Output JSON file path
        """
        if not self.crossing_data:
            print("No crossing data to export")
            return
        
        data = {
            'crossing_records': self.crossing_data,
            'statistics': self.get_statistics(),
            'export_time': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported crossing data to {output_path}")
    
    def reset(self):
        """Reset analyzer"""
        self.entry_times.clear()
        self.exit_times.clear()
        self.crossing_data.clear()
