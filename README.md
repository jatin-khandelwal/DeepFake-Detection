🎭 Deepfake Detection using EfficientNet-B4 & Vision Transformer

An AI-powered Deepfake Detection System that identifies manipulated facial videos using deep learning and computer vision techniques. The project combines EfficientNet-B4 and Vision Transformer (ViT-B/16) with a complete preprocessing pipeline for robust and accurate deepfake classification.

🚀 Features
Deepfake detection using EfficientNet-B4 and Vision Transformer (ViT-B/16)
Video frame extraction and preprocessing
Face detection and cropping using OpenCV
Image quality filtering before inference
Dataset loading and preprocessing for FaceForensics++
Model training, validation, and performance evaluation
Accuracy, ROC-AUC, and classification report generation
Training loss and performance visualization
🛠️ Tech Stack
Python
PyTorch
Torchvision
timm
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
📂 Project Structure
deepfake-detection/
│── dataset/
│── models/
│── checkpoints/
│── results/
│── efficientnet_b4.py
│── dataset_loader.py
│── video_reader.py
│── frame_extractor.py
│── face_detector.py
│── quality_filter.py
│── train.py
│── inference.py
│── requirements.txt
└── README.md
🔍 Workflow
Load videos from the FaceForensics++ dataset.
Extract video frames.
Detect and crop faces.
Filter low-quality frames.
Train EfficientNet-B4 and ViT-B/16 models.
Evaluate using Accuracy, ROC-AUC, and Classification Report.
Predict whether a video is Real or Deepfake.
📊 Model Evaluation
Accuracy
ROC-AUC Score
Precision
Recall
F1-Score
Classification Report
📦 Dataset
FaceForensics++
Supports both Real and Manipulated videos.
🔮 Future Improvements
Real-time webcam deepfake detection
Explainable AI (Grad-CAM) visualization
Mobile and web deployment
ONNX/TensorRT optimization
Ensemble learning with additional CNN architectures
