import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import json
import os
from datetime import datetime


class ModelEvaluator:
    
    
    def __init__(self, model, model_name, device='cuda'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.results = {}
    
    def evaluate(self, test_loader, threshold=0.5):
        """
        Evaluate model on test data
        
        Returns:
            dict: All metrics and predictions
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {self.model_name}")
        print(f"{'='*60}")
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                # Get predictions
                outputs = self.model(inputs)
                probs = outputs.cpu().numpy().flatten()
                
                all_probs.extend(probs)
                all_labels.extend(labels.numpy().flatten())
        
        # Convert to numpy
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
        
        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr) * 100
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Store results
        self.results = {
            'model_name': self.model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist(),
            'labels': all_labels.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        # Print results
        print(f"\n METRICS:")
        print(f"   Accuracy:  {accuracy:.2f}%")
        print(f"   Precision: {precision:.2f}%")
        print(f"   Recall:    {recall:.2f}%")
        print(f"   F1-Score:  {f1:.2f}%")
        print(f"   ROC-AUC:   {roc_auc:.2f}%")
        
        print(f"\n CONFUSION MATRIX:")
        print(f"   TN: {cm[0][0]}  FP: {cm[0][1]}")
        print(f"   FN: {cm[1][0]}  TP: {cm[1][1]}")
        
        print(f"\n CLASSIFICATION REPORT:")
        print(classification_report(all_labels, all_preds, 
                                   target_names=['Real', 'Fake']))
        
        return self.results
    
    def plot_confusion_matrix(self, save_path=None):
        
        cm = np.array(self.results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved: {save_path}")
        plt.close()
    
    def plot_roc_curve(self, save_path=None):
        
        fpr = np.array(self.results['fpr'])
        tpr = np.array(self.results['tpr'])
        roc_auc = self.results['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f}%)')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved: {save_path}")
        plt.close()
    
    def save_results(self, save_path):
      
        json_results = {
            'model_name': self.results['model_name'],
            'accuracy': self.results['accuracy'],
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1_score': self.results['f1_score'],
            'roc_auc': self.results['roc_auc'],
            'confusion_matrix': self.results['confusion_matrix']
        }
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f" Saved: {save_path}")


class MultiModelEvaluator:
    
    
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {}
        self.results = {}
    
    def add_model(self, model, model_name, checkpoint_path=None):
        """Add a model to compare"""
        if checkpoint_path:
            print(f"Loading {model_name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        self.models[model_name] = model.to(self.device)
    
    def evaluate_all(self, test_loader, save_dir='results'):
       
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"MULTI-MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"Models: {list(self.models.keys())}")
        print(f"Save directory: {save_dir}")
        
        for model_name, model in self.models.items():
            evaluator = ModelEvaluator(model, model_name, self.device)
            results = evaluator.evaluate(test_loader)
            
            # Save individual results
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            evaluator.plot_confusion_matrix(
                os.path.join(model_dir, 'confusion_matrix.png'))
            evaluator.plot_roc_curve(
                os.path.join(model_dir, 'roc_curve.png'))
            evaluator.save_results(
                os.path.join(model_dir, 'metrics.json'))
            
            self.results[model_name] = results
        
        # Create comparison plots
        self._plot_comparison_table(save_dir)
        self._plot_combined_roc(save_dir)
        self._save_comparison_json(save_dir)
        
        print(f"\n Evaluation complete! Results saved in: {save_dir}/")
    
    def _plot_comparison_table(self, save_dir):
       
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        
        for model_name, results in self.results.items():
            row = [model_name]
            for metric in metrics:
                row.append(f"{results[metric]:.2f}%")
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(6):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Model Comparison - All Metrics', fontsize=14, weight='bold', pad=20)
        
        save_path = os.path.join(save_dir, 'comparison_table.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.close()
    
    def _plot_combined_roc(self, save_dir):
        """Plot all ROC curves on same graph"""
        plt.figure(figsize=(10, 8))
        
        colors = ['darkorange', 'green', 'blue', 'red', 'purple']
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            fpr = np.array(results['fpr'])
            tpr = np.array(results['tpr'])
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.2f}%)')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models Comparison', fontsize=14, weight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'combined_roc_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.close()
    
    def _save_comparison_json(self, save_dir):
        """Save comparison results to JSON"""
        comparison = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {}
        }
        
        for model_name, results in self.results.items():
            comparison['models'][model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'roc_auc': results['roc_auc'],
                'confusion_matrix': results['confusion_matrix']
            }
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['accuracy'])
        comparison['best_model'] = {
            'name': best_model[0],
            'accuracy': best_model[1]['accuracy']
        }
        
        save_path = os.path.join(save_dir, 'comparison_results.json')
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f" Saved: {save_path}")


#  USAGE EXAMPLE 
if __name__ == "__main__":
    from train_pytorch import load_data_pytorch
    from efficientnet_b4 import EfficientNetB4Deepfake
    from xception import XceptionDeepfake
    from vit import ViTDeepfake
    
    # Configuration
    SPLITS_PATH = 'deepfake_model/dataset/splits'
    RESULTS_DIR = 'deepfake_model/results/frozen_models'
    BATCH_SIZE = 32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load test data
    _, _, test_loader = load_data_pytorch(SPLITS_PATH, batch_size=BATCH_SIZE)
    
    # Create multi-model evaluator
    evaluator = MultiModelEvaluator(device=device)
    
    # Add models
    evaluator.add_model(
        EfficientNetB4Deepfake(pretrained=False, freeze_backbone=False),
        'EfficientNet-B4',
        'deepfake_model/checkpoints/efficientnet_b4/best_model.pth'
    )
    
    evaluator.add_model(
        XceptionDeepfake(pretrained=False, freeze_backbone=False),
        'Xception',
        'deepfake_model/checkpoints/xception/best_model.pth'
    )
    
    evaluator.add_model(
        ViTDeepfake(pretrained=False, freeze_backbone=False),
        'ViT-B/16',
        'deepfake_model/checkpoints/vit/best_model.pth'
    )
    
    # Evaluate all models
    evaluator.evaluate_all(test_loader, save_dir=RESULTS_DIR)
    
    print("\n Done! Check the results folder for:")
    print("   - Individual model metrics")
    print("   - Confusion matrices")
    print("   - ROC curves")
    print("   - Comparison table")
    print("   - Combined ROC plot")