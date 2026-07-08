

import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataSplitter:
   
    
    def __init__(self, random_state: int = 42):
       
        self.random_state = random_state
    
    def split_train_test(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        test_size: float = 0.2,
                        save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        print(f"Splitting dataset: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Maintain real/fake ratio
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"  Real: {np.sum(y_train == 0)}, Fake: {np.sum(y_train == 1)}")
        print(f"Test set: {X_test.shape}")
        print(f"  Real: {np.sum(y_test == 0)}, Fake: {np.sum(y_test == 1)}")
        
        if save_path:
            self._save_splits(X_train, X_test, y_train, y_test, save_path=save_path)
        
        return X_train, X_test, y_train, y_test
    
    def split_train_val_test(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            test_size: float = 0.2,
                            val_size: float = 0.2,
                            save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                       np.ndarray, np.ndarray, np.ndarray]:
        
        print(f"Splitting dataset: train/val/test")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"\nTrain set: {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Real: {np.sum(y_train == 0)}, Fake: {np.sum(y_train == 1)}")
        print(f"Validation set: {X_val.shape} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Real: {np.sum(y_val == 0)}, Fake: {np.sum(y_val == 1)}")
        print(f"Test set: {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  Real: {np.sum(y_test == 0)}, Fake: {np.sum(y_test == 1)}")
        
        if save_path:
            self._save_splits(X_train, X_val, X_test, y_train, y_val, y_test, save_path=save_path, has_val=True)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _save_splits(self, *arrays, save_path: str, has_val: bool = False):
        """Save split datasets"""
        os.makedirs(save_path, exist_ok=True)
        
        if has_val:
            X_train, X_val, X_test, y_train, y_val, y_test = arrays
            
            np.save(os.path.join(save_path, 'X_train.npy'), X_train)
            np.save(os.path.join(save_path, 'X_val.npy'), X_val)
            np.save(os.path.join(save_path, 'X_test.npy'), X_test)
            np.save(os.path.join(save_path, 'y_train.npy'), y_train)
            np.save(os.path.join(save_path, 'y_val.npy'), y_val)
            np.save(os.path.join(save_path, 'y_test.npy'), y_test)
            
            print(f"\n Saved to: {save_path}/")
            print(f"   X_train.npy, X_val.npy, X_test.npy")
            print(f"   y_train.npy, y_val.npy, y_test.npy")
        else:
            X_train, X_test, y_train, y_test = arrays
            
            np.save(os.path.join(save_path, 'X_train.npy'), X_train)
            np.save(os.path.join(save_path, 'X_test.npy'), X_test)
            np.save(os.path.join(save_path, 'y_train.npy'), y_train)
            np.save(os.path.join(save_path, 'y_test.npy'), y_test)
            
            print(f"\n Saved to: {save_path}/")
            print(f"   X_train.npy, X_test.npy")
            print(f"   y_train.npy, y_test.npy")
    
    @staticmethod
    def load_splits(load_path: str, has_val: bool = False):
        
        if has_val:
            X_train = np.load(os.path.join(load_path, 'X_train.npy'))
            X_val = np.load(os.path.join(load_path, 'X_val.npy'))
            X_test = np.load(os.path.join(load_path, 'X_test.npy'))
            y_train = np.load(os.path.join(load_path, 'y_train.npy'))
            y_val = np.load(os.path.join(load_path, 'y_val.npy'))
            y_test = np.load(os.path.join(load_path, 'y_test.npy'))
            
            print(f" Loaded from: {load_path}/")
            print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train = np.load(os.path.join(load_path, 'X_train.npy'))
            X_test = np.load(os.path.join(load_path, 'X_test.npy'))
            y_train = np.load(os.path.join(load_path, 'y_train.npy'))
            y_test = np.load(os.path.join(load_path, 'y_test.npy'))
            
            print(f" Loaded from: {load_path}/")
            print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test


#  USAGE EXAMPLE
if __name__ == "__main__":
    print("="*70)
    print("DATA SPLITTER")
    print("="*70)
    
    # Load preprocessed dataset
    data_path = 'deepfake_model/dataset/processed_dataset/ff++'
    
    print(f"\nLoading data from: {data_path}")
    X = np.load(os.path.join(data_path, 'X.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    
    print(f"Loaded: {X.shape}, Labels: {y.shape}")
    print(f"Real: {np.sum(y == 0)}, Fake: {np.sum(y == 1)}")
    
    
    splitter = DataSplitter(random_state=42)
    
   
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT (80/20)")
    print("="*70)
    
    X_train, X_test, y_train, y_test = splitter.split_train_test(
        X, y,
        test_size=0.2,
        save_path='deepfake_model/dataset/splits'
    )
    
    print("\n Split complete!")