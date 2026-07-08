"""
dataset_builder.py
Build dataset from videos by combining all preprocessing steps
Optimized for DFDC, FaceForensics++, and Celeb-DF datasets
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from dataset_loader import FaceForensicsLoader
from video_reader import VideoReader
from frame_extractor import FrameExtractor
from face_detector import FaceDetector
from quality_filter import QualityFilter


class DatasetBuilder:
    
    
    def __init__(self,
                 face_detector_method: str = 'haar',
                 blur_threshold: float = 50.0,
                 min_brightness: float = 30.0,
                 max_brightness: float = 230.0,
                 min_contrast: float = 30.0,
                 min_size: int = 60,
                 extraction_fps: float = 1.0,
                 target_size: Tuple[int, int] = (224, 224),
                 padding: float = 0.2):
     
        self.face_detector = FaceDetector(method=face_detector_method)
        
        # Initialize quality filter with relaxed thresholds for compressed videos
        self.quality_filter = QualityFilter(
            blur_threshold=blur_threshold,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            min_contrast=min_contrast,
            min_size=min_size
        )
        
        self.extraction_fps = extraction_fps
        self.target_size = target_size
        self.padding = padding
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_faces': 0,
            'failed_reasons': []
        }
    
    def process_video(self, video_path: str,
                      max_faces: int = None,
                      verbose: bool = True) -> List[np.ndarray]:
       
        if verbose:
            print(f"Processing: {os.path.basename(video_path)}")
        
        faces = []
        self.stats['total_videos'] += 1
        
        # Check if file exists
        if not os.path.exists(video_path):
            if verbose:
                print(f"  ❌ ERROR: Video file not found")
            self.stats['failed_videos'] += 1
            self.stats['failed_reasons'].append(('not_found', video_path))
            return []
        
        try:
            with VideoReader(video_path) as vr:
                # Get video metadata
                if verbose:
                    metadata = vr.get_metadata()
                    print(f"  Duration: {metadata['duration']:.2f}s, "
                          f"FPS: {metadata['fps']:.1f}, "
                          f"Resolution: {metadata['width']}x{metadata['height']}")
                
                # Extract frames
                extractor = FrameExtractor(vr)
                frames = extractor.extract_by_fps(self.extraction_fps)
                
                if verbose:
                    print(f"  Extracted {len(frames)} frames")
                
                if len(frames) == 0:
                    if verbose:
                        print(f"  ⚠️  WARNING: No frames extracted")
                    self.stats['failed_videos'] += 1
                    self.stats['failed_reasons'].append(('no_frames', video_path))
                    return []
                
                # Process each frame
                frames_with_faces = 0
                for frame_num, frame in frames:
                    # Detect and crop faces
                    detected_faces = self.face_detector.crop_faces(
                        frame, 
                        padding=self.padding
                    )
                    
                    if len(detected_faces) > 0:
                        frames_with_faces += 1
                    
                    # Filter by quality and preprocess
                    for face in detected_faces:
                        if self.quality_filter.passes_quality_check(face):
                            # Resize to target size
                            face_resized = cv2.resize(
                                face, 
                                self.target_size,
                                interpolation=cv2.INTER_LINEAR
                            )
                            
                            # Convert BGR to RGB
                            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                            
                            # Normalize to [0, 1]
                            face_normalized = face_rgb.astype('float32') / 255.0
                            
                            faces.append(face_normalized)
                            
                            # Check max limit
                            if max_faces and len(faces) >= max_faces:
                                if verbose:
                                    print(f"  ✓ Reached max faces limit: {max_faces}")
                                self.stats['successful_videos'] += 1
                                self.stats['total_faces'] += len(faces)
                                return faces
                
                if verbose:
                    print(f"  ✓ Collected {len(faces)} quality faces "
                          f"from {frames_with_faces}/{len(frames)} frames")
                
                if len(faces) == 0:
                    if verbose:
                        print(f"    WARNING: No quality faces found")
                    self.stats['failed_videos'] += 1
                    self.stats['failed_reasons'].append(('no_quality_faces', video_path))
                else:
                    self.stats['successful_videos'] += 1
                    self.stats['total_faces'] += len(faces)
        
        except FileNotFoundError:
            if verbose:
                print(f"   ERROR: Video file not found")
            self.stats['failed_videos'] += 1
            self.stats['failed_reasons'].append(('file_not_found', video_path))
            return []
        
        except RuntimeError as e:
            if verbose:
                print(f"   ERROR: Cannot open video - {str(e)}")
            self.stats['failed_videos'] += 1
            self.stats['failed_reasons'].append(('open_failed', video_path))
            return []
        
        except cv2.error as e:
            if verbose:
                print(f"   ERROR: OpenCV error - {str(e)}")
            self.stats['failed_videos'] += 1
            self.stats['failed_reasons'].append(('opencv_error', video_path))
            return []
        
        except Exception as e:
            if verbose:
                print(f"   ERROR: Unexpected error - {type(e).__name__}: {str(e)}")
            self.stats['failed_videos'] += 1
            self.stats['failed_reasons'].append(('unexpected', video_path))
            return []
        
        return faces
    
    def process_dataset(self,
                    video_paths: List[str],
                    labels: List[int],
                    max_faces_per_video: int = 50,
                    save_path: Optional[str] = None,
                    verbose: bool = True,
                    batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:  # ← Add batch_size
       
        if len(video_paths) != len(labels):
            raise ValueError(f"video_paths ({len(video_paths)}) and labels ({len(labels)}) must have same length")
        
        # Create save directory first
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        all_faces = []
        all_labels = []
        batch_count = 0
        
        # Reset statistics
        self.stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_faces': 0,
            'failed_reasons': []
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING {len(video_paths)} VIDEOS")
            print(f"Saving in batches of {batch_size} videos to avoid memory issues")
            print(f"{'='*70}\n")
        
        for idx, (video_path, label) in enumerate(zip(video_paths, labels)):
            if verbose:
                print(f"[{idx+1}/{len(video_paths)}] ", end='')
            
            try:
                faces = self.process_video(
                    video_path,
                    max_faces=max_faces_per_video,
                    verbose=verbose
                )
                
                if faces:
                    all_faces.extend(faces)
                    all_labels.extend([label] * len(faces))
            
            except KeyboardInterrupt:
                if verbose:
                    print(f"\n\n  Processing interrupted by user")
                break
            
            except Exception as e:
                if verbose:
                    print(f"   CRITICAL ERROR: {type(e).__name__}: {str(e)}")
                continue
            
            #  SAVE BATCH EVERY N VIDEOS
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(video_paths):
                if len(all_faces) > 0 and save_path:
                    batch_count += 1
                    
                    X_batch = np.array(all_faces, dtype='float32')
                    y_batch = np.array(all_labels, dtype='int32')
                    
                    batch_file = os.path.join(save_path, f'batch_{batch_count:03d}.npz')
                    np.savez_compressed(batch_file, X=X_batch, y=y_batch)
                    
                    if verbose:
                        print(f"\n💾 Saved batch {batch_count}: {X_batch.shape} ({X_batch.nbytes / (1024**2):.1f} MB)")
                    
                    # Clear memory
                    all_faces = []
                    all_labels = []
        
        #  MERGE ALL BATCHES
        if verbose:
            print(f"\n{'='*70}")
            print(f"MERGING {batch_count} BATCHES")
            print(f"{'='*70}")
        
        if save_path and batch_count > 0:
            X_list = []
            y_list = []
            
            for i in range(1, batch_count + 1):
                batch_file = os.path.join(save_path, f'batch_{i:03d}.npz')
                data = np.load(batch_file)
                X_list.append(data['X'])
                y_list.append(data['y'])
                data.close()
                
                if verbose:
                    print(f"Loaded batch {i}/{batch_count}")
            
            X = np.concatenate(X_list)
            y = np.concatenate(y_list)
            
            # Delete batch files to save space
            import time
            for i in range(1, batch_count + 1):
                batch_file = os.path.join(save_path, f'batch_{i:03d}.npz')
                try:
                    os.remove(batch_file)
                except PermissionError:
                    time.sleep(0.1)  # Wait a bit
                    os.remove(batch_file)  # Try again
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"DATASET PROCESSING COMPLETE")
                print(f"{'='*70}")
                print(f"Videos processed: {self.stats['total_videos']}")
                print(f"  ✓ Successful: {self.stats['successful_videos']}")
                print(f"  ✗ Failed: {self.stats['failed_videos']}")
                print(f"\nFaces extracted: {len(X)}")
                print(f"  Real faces (label=0): {np.sum(y == 0)}")
                print(f"  Fake faces (label=1): {np.sum(y == 1)}")
                print(f"\nDataset shape: {X.shape}")
                print(f"Labels shape: {y.shape}")
                print(f"Memory usage: {X.nbytes / (1024**2):.2f} MB")
                
                if self.stats['failed_videos'] > 0:
                    print(f"\nFailure breakdown:")
                    failure_counts = {}
                    for reason, _ in self.stats['failed_reasons']:
                        failure_counts[reason] = failure_counts.get(reason, 0) + 1
                    for reason, count in failure_counts.items():
                        print(f"  {reason}: {count}")
                
                print(f"{'='*70}\n")
            
            # Save metadata
            self.save_dataset(X, y, save_path, verbose=False)  # Just save metadata
            
            return X, y
        
        return np.array([]), np.array([])
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray,
                    save_path: str, verbose: bool = True):
       
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save arrays
        X_path = os.path.join(save_path, 'X.npy')
        y_path = os.path.join(save_path, 'y.npy')
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        if verbose:
            print(f" Dataset saved to: {save_path}")
            print(f"  X.npy: {X.shape} ({os.path.getsize(X_path) / (1024**2):.2f} MB)")
            print(f"  y.npy: {y.shape} ({os.path.getsize(y_path) / (1024**2):.2f} MB)")
        
        # Save metadata
        metadata = {
            'shape': X.shape,
            'num_samples': len(X),
            'num_real': int(np.sum(y == 0)),
            'num_fake': int(np.sum(y == 1)),
            'target_size': self.target_size,
            'extraction_fps': self.extraction_fps,
            'face_detector': self.face_detector.method,
            'statistics': self.stats
        }
        
        import json
        metadata_path = os.path.join(save_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            print(f"  metadata.json: Dataset info saved")
    
    @staticmethod
    def load_dataset(load_path: str, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
       
        X_path = os.path.join(load_path, 'X.npy')
        y_path = os.path.join(load_path, 'y.npy')
        
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Dataset files not found in {load_path}")
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        if verbose:
            print(f" Dataset loaded from: {load_path}")
            print(f"  X: {X.shape}")
            print(f"  y: {y.shape}")
            print(f"  Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")
        
        return X, y
    
    def get_statistics(self) -> Dict:
        
        return self.stats.copy()
    
    def reset_statistics(self):
        
        self.stats = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'total_faces': 0,
            'failed_reasons': []
        }


# ============ USAGE EXAMPLE ============
# ============ TEST EXAMPLE (20 videos, batch size 5) ============
if __name__ == "__main__":
    print("="*70)
    print("FULL DATASET PROCESSING")
    print("="*70)
    
    builder = DatasetBuilder(
        face_detector_method='haar',
        blur_threshold=50.0,
        extraction_fps=1.0,
        target_size=(224, 224),
        padding=0.2
    )
    
    
    loader = FaceForensicsLoader('deepfake_model/dataset/raw_dataset/ff++')
    video_paths, labels = loader.load_videos()
    
    print(f"\n Processing {len(video_paths)} videos")
    print(f"   Real: {labels.count(0)}, Fake: {labels.count(1)}\n")
    
    X, y = builder.process_dataset(
        video_paths=video_paths,
        labels=labels,
        max_faces_per_video=50,
        save_path='deepfake_model/dataset/processed_dataset/ff++',
        verbose=True,
        batch_size=20
    )
    
    if len(X) > 0:
        print(f"\n COMPLETE!")
        print(f"Shape: {X.shape}")
        print(f"Real: {np.sum(y==0)}, Fake: {np.sum(y==1)}")