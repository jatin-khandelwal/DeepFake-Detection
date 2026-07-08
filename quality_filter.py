import cv2
import numpy as np
from typing import Dict, Optional


class QualityFilter:
    def __init__(self,
                 blur_threshold: float = 50.0,  
                 min_brightness: float = 30.0, 
                 max_brightness: float = 230.0,
                 min_contrast: float = 30.0,    
                 min_size: int = 60):         
       
        
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_size = min_size
    
    def calculate_blur(self, image: np.ndarray) -> float:
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.mean(gray)[0]
    
    def calculate_contrast(self, image: np.ndarray) -> float:
       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        return max_val - min_val
    
    def get_metrics(self, image: np.ndarray) -> Dict[str, float]:
       
        return {
            'blur': self.calculate_blur(image),
            'brightness': self.calculate_brightness(image),
            'contrast': self.calculate_contrast(image),
            'height': image.shape[0],
            'width': image.shape[1]
        }
    
    def is_sharp(self, image: np.ndarray) -> bool:
        """Check if image is sharp enough"""
        return self.calculate_blur(image) >= self.blur_threshold
    
    def is_well_lit(self, image: np.ndarray) -> bool:
        """Check if image has good lighting"""
        brightness = self.calculate_brightness(image)
        return self.min_brightness <= brightness <= self.max_brightness
    
    def has_good_contrast(self, image: np.ndarray) -> bool:
        """Check if image has sufficient contrast"""
        return self.calculate_contrast(image) >= self.min_contrast
    
    def is_large_enough(self, image: np.ndarray) -> bool:
        """Check if image is large enough"""
        h, w = image.shape[:2]
        return min(h, w) >= self.min_size
    
    def passes_quality_check(self, image: np.ndarray,
                            verbose: bool = False) -> bool:
        
        checks = {
            'sharp': self.is_sharp(image),
            'well_lit': self.is_well_lit(image),
            'good_contrast': self.has_good_contrast(image),
            'large_enough': self.is_large_enough(image)
        }
        
        if verbose:
            metrics = self.get_metrics(image)
            print(f"Quality Metrics:")
            print(f"  Blur: {metrics['blur']:.2f} (threshold: {self.blur_threshold})")
            print(f"  Brightness: {metrics['brightness']:.2f} ({self.min_brightness}-{self.max_brightness})")
            print(f"  Contrast: {metrics['contrast']:.2f} (min: {self.min_contrast})")
            print(f"  Size: {metrics['width']}x{metrics['height']} (min: {self.min_size})")
            print(f"Checks: {checks}")
        
        return all(checks.values())
    
    def filter_faces(self, faces: list) -> list:
        
        return [face for face in faces if self.passes_quality_check(face)]
    
    def get_quality_score(self, image: np.ndarray) -> float:
        
        blur = min(self.calculate_blur(image) / self.blur_threshold, 1.0)
        
        brightness = self.calculate_brightness(image)
        bright_score = 1.0 - abs(brightness - 127.5) / 127.5
        
        contrast = min(self.calculate_contrast(image) / 255.0, 1.0)
        
        h, w = image.shape[:2]
        size_score = min(min(h, w) / self.min_size, 1.0)
        
        # Weighted average
        return (blur * 0.4 + bright_score * 0.3 + contrast * 0.2 + size_score * 0.1)


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    qf = QualityFilter(
        blur_threshold=50.0,
        min_brightness=30.0,
        max_brightness=230.0,
        min_contrast=30.0,
        min_size=60
    )
    
    # Test on image
    img = cv2.imread('test_face.jpg')
    if img is not None:
        # Check quality
        passes = qf.passes_quality_check(img, verbose=True)
        print(f"\nPasses quality check: {passes}")
        
        # Get quality score
        score = qf.get_quality_score(img)
        print(f"Quality score: {score:.2f}")