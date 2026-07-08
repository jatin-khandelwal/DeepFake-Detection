import torch
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepfake_model.model.efficientnet_b4 import EfficientNetB4Deepfake
from deepfake_model.model.xception import XceptionDeepfake
from deepfake_model.model.vit import ViTDeepfake

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeepfakePredictor:

    def __init__(self, checkpoint_paths):

        self.models = {}

        print("Loading models...")

        # Load models
        self.models["efficientnet"] = EfficientNetB4Deepfake(pretrained=False)
        self.models["efficientnet"].load_state_dict(
            torch.load(checkpoint_paths["efficientnet"], map_location=DEVICE)["model_state_dict"]
        )

        self.models["xception"] = XceptionDeepfake(pretrained=False)
        self.models["xception"].load_state_dict(
            torch.load(checkpoint_paths["xception"], map_location=DEVICE)["model_state_dict"]
        )

        self.models["vit"] = ViTDeepfake(pretrained=False)
        self.models["vit"].load_state_dict(
            torch.load(checkpoint_paths["vit"], map_location=DEVICE)["model_state_dict"]
        )

        # Move to device
        for model in self.models.values():
            model.to(DEVICE)
            model.eval()

    def preprocess(self, face):
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype("float32") / 255.0
        face = np.transpose(face, (2, 0, 1))

        face = torch.tensor(face).float()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(DEVICE)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(DEVICE)

        face = (face - mean) / std
        face = face.unsqueeze(0)

        return face.to(DEVICE)

    def predict_all(self, face):

        tensor = self.preprocess(face)

        results = {}
        probs_list = []

        with torch.no_grad():
            for name, model in self.models.items():

                output = model(tensor)

                # Handle binary or multi-class
                if output.shape[-1] == 1:
                    prob = torch.sigmoid(output).item()
                else:
                    prob = torch.softmax(output, dim=1)[0][1].item()

                results[name] = prob
                probs_list.append(prob)

        # 🔥 Stable ensemble
        results["ensemble_mean"] = float(np.mean(probs_list))
        results["ensemble_median"] = float(np.median(probs_list))

        # Use median as final
        results["ensemble"] = results["ensemble_median"]

        return results
