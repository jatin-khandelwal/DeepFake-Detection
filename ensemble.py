
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


class DeepfakeEnsemble:
   

    def __init__(self, models_dict, device='cuda'):
       
        self.device = device
        self.models = {}
        self.weights = {}

        # Load models and normalize weights
        total_weight = sum(w for _, w in models_dict.values())
        for name, (model, weight) in models_dict.items():
            self.models[name] = model.to(device)
            self.weights[name] = weight / total_weight

        print(f" Ensemble weights:")
        for name, w in self.weights.items():
            print(f"   {name}: {w:.2f}")

    def predict(self, dataloader, threshold=0.5):
       
        for model in self.models.values():
            model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)

                # Weighted sum of all model predictions
                ensemble_probs = torch.zeros(inputs.size(0), 1).to(self.device)
                for name, model in self.models.items():
                    probs = model(inputs)   # (B, 1)
                    ensemble_probs += self.weights[name] * probs

                all_probs.extend(ensemble_probs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs > threshold).astype(int)

        return all_preds, all_probs, all_labels

    def evaluate(self, dataloader, threshold=0.5):
        """Full evaluation with metrics"""
        print("\n" + "="*50)
        print("ENSEMBLE EVALUATION")
        print("="*50)

        preds, probs, labels = self.predict(dataloader, threshold)

        acc = accuracy_score(labels, preds) * 100
        auc = roc_auc_score(labels, probs) * 100

        print(f"Accuracy:  {acc:.2f}%")
        print(f"ROC-AUC:   {auc:.2f}%")
        print("\nClassification Report:")
        print(classification_report(labels, preds,
                                    target_names=['Real', 'Fake']))
        return acc, auc

    def find_best_weights(self, val_loader, steps=6):
        
        print("\n Searching for best ensemble weights...")
        best_acc = 0
        best_weights = self.weights.copy()
        model_names = list(self.models.keys())

        w_range = np.linspace(0, 1, steps)

        for w0 in w_range:
            for w1 in w_range:
                w2 = 1.0 - w0 - w1
                if w2 < 0:
                    continue

                self.weights[model_names[0]] = w0
                self.weights[model_names[1]] = w1
                self.weights[model_names[2]] = w2

                preds, _, labels = self.predict(val_loader)
                acc = accuracy_score(labels, preds) * 100

                if acc > best_acc:
                    best_acc = acc
                    best_weights = {
                        model_names[0]: w0,
                        model_names[1]: w1,
                        model_names[2]: w2
                    }
                    print(f"  New best: {model_names[0]}={w0:.2f}, "
                          f"{model_names[1]}={w1:.2f}, "
                          f"{model_names[2]}={w2:.2f} → Acc={acc:.2f}%")

        # Set best weights
        self.weights = best_weights
        print(f"\n Best Val Acc: {best_acc:.2f}%")
        return best_weights


def load_ensemble_from_checkpoints(checkpoint_paths, device='cuda'):
   
    from efficientnet_b4 import EfficientNetB4Deepfake
    from xception import XceptionDeepfake
    from vit import ViTDeepfake

    model_classes = {
        'efficientnet': EfficientNetB4Deepfake,
        'xception': XceptionDeepfake,
        'vit': ViTDeepfake
    }

    models_dict = {}
    default_weights = {'efficientnet': 0.35, 'xception': 0.35, 'vit': 0.30}

    for name, path in checkpoint_paths.items():
        print(f"\nLoading {name}...")
        model = model_classes[name](pretrained=False, freeze_backbone=False)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded from epoch {checkpoint['epoch']+1}")
        models_dict[name] = (model, default_weights[name])

    return DeepfakeEnsemble(models_dict, device=device)


# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    from train_pytorch import load_data_pytorch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    SPLITS_PATH = 'deepfake_model/dataset/splits'

    checkpoint_paths = {
        'efficientnet': 'deepfake_model/checkpoints/efficientnet_b4/best_model.pth',
        'xception':     'deepfake_model/checkpoints/xception/best_model.pth',
        'vit':          'deepfake_model/checkpoints/vit/best_model.pth',
    }

    _, val_loader, test_loader = load_data_pytorch(SPLITS_PATH, batch_size=32)

    ensemble = load_ensemble_from_checkpoints(checkpoint_paths, device)
    ensemble.find_best_weights(val_loader)
    ensemble.evaluate(test_loader)