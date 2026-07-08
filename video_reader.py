import cv2
import os
from typing import Dict, Optional
import numpy as np


class VideoReader:
    
    
    def __init__(self, video_path: str):
       
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load video metadata"""
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def get_metadata(self) -> Dict:
        
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'size_mb': os.path.getsize(self.video_path) / (1024 * 1024)
        }
    
    def read_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
       
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def read_frames(self, start_frame: int = 0, 
                    end_frame: Optional[int] = None,
                    step: int = 1):
       
        if end_frame is None:
            end_frame = self.frame_count
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame, step):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_num, frame
            
            # Skip frames according to step
            if step > 1:
                for _ in range(step - 1):
                    self.cap.read()
    
    def get_current_frame_number(self) -> int:
        """Get current frame position"""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def reset(self):
        """Reset to beginning of video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self):
        """Release video capture"""
        if self.cap.isOpened():
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


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    with VideoReader('Hollow Knight Silksong 2025-12-27 23-14-16.mp4') as vr:
        # Print metadata
        metadata = vr.get_metadata()
        print("Video Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Read first 10 frames
        print("\nReading first 10 frames...")
        for frame_num, frame in vr.read_frames(0, 500, step=1):
            print(f"Frame {frame_num}: {frame.shape}")