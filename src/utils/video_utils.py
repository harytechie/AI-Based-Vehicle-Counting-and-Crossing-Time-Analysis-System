"""
Video processing utilities
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from pathlib import Path


class VideoProcessor:
    """Handles video loading, frame extraction, and preprocessing"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        
        self._load_video()
    
    def _load_video(self):
        """Load video and extract metadata"""
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        
        # Extract video metadata
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
    
    def get_video_info(self) -> dict:
        """
        Get video metadata
        
        Returns:
            Dictionary containing video information
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'duration_formatted': f"{int(self.duration // 60)}:{int(self.duration % 60):02d}"
        }
    
    def extract_frames(self, skip_frames: int = 0) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video
        
        Args:
            skip_frames: Process every Nth frame (0 = process all frames)
            
        Yields:
            Tuple of (frame_number, frame)
        """
        if self.cap is None:
            raise ValueError("Video not loaded")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Skip frames if specified
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            yield frame_count, frame
            frame_count += 1
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by number
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None:
            raise ValueError("Video not loaded")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def preprocess_frame(self, frame: np.ndarray, 
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess frame (resize, normalize, etc.)
        
        Args:
            frame: Input frame
            target_size: Target size (width, height) for resizing
            
        Returns:
            Preprocessed frame
        """
        processed = frame.copy()
        
        # Resize if target size specified
        if target_size is not None:
            processed = cv2.resize(processed, target_size)
        
        return processed
    
    def frame_to_timestamp(self, frame_number: int) -> float:
        """
        Convert frame number to timestamp in seconds
        
        Args:
            frame_number: Frame number
            
        Returns:
            Timestamp in seconds
        """
        if self.fps == 0:
            return 0
        return frame_number / self.fps
    
    def timestamp_to_frame(self, timestamp: float) -> int:
        """
        Convert timestamp to frame number
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            Frame number
        """
        return int(timestamp * self.fps)
    
    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __del__(self):
        """Destructor"""
        self.release()


def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int],
                              font_scale: float = 0.6, thickness: int = 2,
                              text_color: Tuple[int, int, int] = (255, 255, 255),
                              bg_color: Tuple[int, int, int] = (0, 0, 0),
                              padding: int = 5) -> np.ndarray:
    """
    Draw text with background rectangle on frame
    
    Args:
        frame: Input frame
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
        padding: Padding around text
        
    Returns:
        Frame with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(frame,
                 (x - padding, y - text_height - padding),
                 (x + text_width + padding, y + baseline + padding),
                 bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)
    
    return frame
