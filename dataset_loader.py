"""
dataset_loaders.py
Load video paths and labels from different dataset formats
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict




class FaceForensicsLoader:
   
    
    def __init__(self, dataset_path: str):
        
        self.dataset_path = dataset_path
        
        # Try to find real folder (handles different naming)
        self.real_folder = None
        self.fake_folder = None
        
        for folder in os.listdir(dataset_path):
            folder_lower = folder.lower()
            full_path = os.path.join(dataset_path, folder)
            
            if not os.path.isdir(full_path):
                continue
                
            if folder_lower in ['real', 'original', 'real_videos']:
                self.real_folder = full_path
            elif folder_lower in ['fake', 'manipulated', 'fake_videos', 'synthesis']:
                self.fake_folder = full_path
    
    def load_videos(self, max_real: int = None, max_fake: int = None) -> Tuple[List[str], List[int]]:
        
        video_paths = []
        labels = []
        
        # Load REAL videos
        if self.real_folder and os.path.exists(self.real_folder):
            real_videos = [
                os.path.join(self.real_folder, f)
                for f in os.listdir(self.real_folder)
                if f.endswith(('.mp4', '.avi', '.mov'))
            ]
            
            if max_real:
                real_videos = real_videos[:max_real]
            
            video_paths.extend(real_videos)
            labels.extend([0] * len(real_videos))
        else:
            print(f"  Warning: Real folder not found in {self.dataset_path}")
        
        # Load FAKE videos
        if self.fake_folder and os.path.exists(self.fake_folder):
            fake_videos = [
                os.path.join(self.fake_folder, f)
                for f in os.listdir(self.fake_folder)
                if f.endswith(('.mp4', '.avi', '.mov'))
            ]
            
            if max_fake:
                fake_videos = fake_videos[:max_fake]
            
            video_paths.extend(fake_videos)
            labels.extend([1] * len(fake_videos))
        else:
            print(f"  Warning: Fake folder not found in {self.dataset_path}")
        
        print(f"FaceForensics Dataset: Loaded {len(video_paths)} videos")
        print(f"  Real: {labels.count(0)}, Fake: {labels.count(1)}")
        
        return video_paths, labels
# USAGE EXAMPLE
if __name__ == "__main__":
    # FaceForensics++ from Kaggle
    ff = FaceForensicsLoader('deepfake_model/dataset/raw_dataset/ff++')
    videos, labels = ff.load_videos()
    
    print(f"Found {len(videos)} videos")
    print(f"Real: {labels.count(0)}, Fake: {labels.count(1)}")
    
    