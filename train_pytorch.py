
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json

import matplotlib.pyplot as plt

from efficientnet_b4 import EfficientNetB4Deepfake


class DeepfakeTrainer:
    """
    Training pipeline for deepfake detection models
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        learning_rate=1e-4,
        batch_size=32,
        epochs=20,
        save_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross Entropy
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)  # (B, 1)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader=None):
        
        print("="*60)
        print("TRAINING START")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Train batches: {len(train_loader)}")
        if val_loader:
            print(f"Val batches: {len(val_loader)}")
        print("="*60)
        
        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth', epoch, val_loss, val_acc)
                    print(f"  ✅ Best model saved!")
            else:
                # No validation - save based on train loss
                if train_loss < self.best_val_loss:
                    self.best_val_loss = train_loss
                    self.save_checkpoint('best_model.pth', epoch, train_loss, train_acc)
                    print(f"  ✅ Best model saved!")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        # Save final model and history
        self.save_checkpoint('final_model.pth', self.epochs - 1)
        self.save_history()
        self.plot_history()
    
    def save_checkpoint(self, filename, epoch, val_loss=None, val_acc=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """
        Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        start_epoch = checkpoint['epoch'] + 1
        
        print(f" Resumed from epoch {start_epoch}")
        return start_epoch


    def save_history(self):
        """Save training history as JSON"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f" History saved: {history_path}")
    
    def plot_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        if self.history['val_loss']:
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        if self.history['val_acc']:
            ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=100)
        plt.show()
        print(f" Training curves saved: {plot_path}")
        plt.close()


def load_data_pytorch(splits_path, batch_size=32, val_split=0.2):
   
    import numpy as np
    
    print("Loading data from .npy files...")
    
    # Load numpy arrays
    X_train_np = np.load(os.path.join(splits_path, 'X_train.npy'))
    y_train_np = np.load(os.path.join(splits_path, 'y_train.npy'))
    X_test_np = np.load(os.path.join(splits_path, 'X_test.npy'))
    y_test_np = np.load(os.path.join(splits_path, 'y_test.npy'))
    
    print(f"  X_train: {X_train_np.shape}")
    print(f"  y_train: {y_train_np.shape}")
    print(f"  X_test: {X_test_np.shape}")
    print(f"  y_test: {y_test_np.shape}")
    
    # Convert to PyTorch tensors
    # NHWC (numpy) → NCHW (PyTorch)
    X_train = torch.from_numpy(X_train_np).permute(0, 3, 1, 2).float()
    y_train = torch.from_numpy(y_train_np).long()
    X_test = torch.from_numpy(X_test_np).permute(0, 3, 1, 2).float()
    y_test = torch.from_numpy(y_test_np).long()
    
    print(f"  Converted to PyTorch tensors (NCHW format)")
    # normalize using ImageNet mean/std:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std    
    
    # Split train into train + validation
    n_val = int(len(X_train) * val_split)
    n_train = len(X_train) - n_val
    
    indices = torch.randperm(len(X_train))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    
    print(f"\nAfter train/val split:")
    print(f"  Train: {X_train_split.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # Configuration
    SPLITS_PATH = 'deepfake_model/dataset/splits'  
    CHECKPOINT_DIR = 'deepfake_model/checkpoints/efficientnet'
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Device-'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data_pytorch(
        SPLITS_PATH,  # ← Changed
        batch_size=BATCH_SIZE,
        val_split=0.2
    )
    
    # Create model
    print("\nCreating EfficientNet-B4 model...")
    model = EfficientNetB4Deepfake(pretrained=True, freeze_backbone=True)
    
    # Create trainer
    trainer = DeepfakeTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        save_dir=CHECKPOINT_DIR
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n Training finished!")
    print(f" Checkpoints saved in: {CHECKPOINT_DIR}/")