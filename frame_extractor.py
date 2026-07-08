import cv2
import numpy as np
from typing import List, Tuple, Optional
from video_reader import VideoReader


class FrameExtractor:
    
    
    def __init__(self, video_reader: VideoReader):
        
        self.video_reader = video_reader
    
    def extract_uniform(self, num_frames: int) -> List[Tuple[int, np.ndarray]]:
        
        total_frames = self.video_reader.frame_count
        
        if num_frames >= total_frames:
            # Extract all frames
            indices = list(range(total_frames))
        else:
            # Uniformly sample
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = self.video_reader.read_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        
        return frames
    
    def extract_by_interval(self, interval: int) -> List[Tuple[int, np.ndarray]]:
       
        frames = []
        for frame_num, frame in self.video_reader.read_frames(step=interval):
            frames.append((frame_num, frame))
        
        return frames
    
    def extract_by_fps(self, target_fps: float) -> List[Tuple[int, np.ndarray]]:
        
        video_fps = self.video_reader.fps
        
        if target_fps >= video_fps:
            # Extract all frames
            interval = 1
        else:
            # Calculate interval
            interval = int(video_fps / target_fps)
        
        return self.extract_by_interval(interval)
    
    def extract_by_time(self, time_interval: float) -> List[Tuple[int, np.ndarray]]:
        
        fps = self.video_reader.fps
        frame_interval = int(time_interval * fps)
        
        return self.extract_by_interval(max(1, frame_interval))
    
    def extract_keyframes(self, threshold: float = 30.0) -> List[Tuple[int, np.ndarray]]:
        
        frames = []
        prev_frame = None
        
        for frame_num, frame in self.video_reader.read_frames():
            if prev_frame is None:
                # Always include first frame
                frames.append((frame_num, frame))
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue
            
            # Calculate frame difference
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, curr_gray)
            mean_diff = np.mean(diff)
            
            # If significant change, consider it a keyframe
            if mean_diff > threshold:
                frames.append((frame_num, frame))
                prev_frame = curr_gray
        
        return frames
    
    def extract_custom(self, frame_indices: List[int]) -> List[Tuple[int, np.ndarray]]:
       
        frames = []
        for idx in frame_indices:
            frame = self.video_reader.read_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        
        return frames
    
def extract_adaptive(self, 
                    target_frames: int = 10,
                    min_interval: int = 5) -> List[Tuple[int, np.ndarray]]:
    
    total_frames = self.video_reader.frame_count
    
    if total_frames <= target_frames * min_interval:
        # Short video - extract uniformly
        return self.extract_uniform(target_frames)
    else:
        # Long video - use interval
        interval = max(min_interval, total_frames // target_frames)
        return self.extract_by_interval(interval)

# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    with VideoReader('Hollow Knight Silksong 2025-12-27 23-14-16.mp4') as vr:
        extractor = FrameExtractor(vr)
        
        # Extract 10 uniform frames
        frames = extractor.extract_uniform(10)
        print(f"Extracted {len(frames)} uniform frames")
        
        # Extract 1 frame per second
        frames = extractor.extract_by_fps(1.0)
        print(f"Extracted {len(frames)} frames at 1 FPS")
        
        # Extract every 30 frames
        frames = extractor.extract_by_interval(30)
        print(f"Extracted {len(frames)} frames (every 30th)")
        
        # Extract keyframes
        frames = extractor.extract_keyframes(threshold=30.0)
        print(f"Extracted {len(frames)} keyframes")