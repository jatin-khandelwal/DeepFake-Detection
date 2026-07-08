import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:

    def __init__(self, method: str = 'haar', min_confidence: float = 0.9):
        self.method = method.lower()
        self.min_confidence = min_confidence

        if self.method == 'haar':
            self._init_haar()
        elif self.method == 'mtcnn':
            self._init_mtcnn()
        elif self.method == 'retinaface':
            self._init_retinaface()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _init_haar(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

    def _init_mtcnn(self):
        from mtcnn import MTCNN
        self.detector = MTCNN()

    def _init_retinaface(self):
        from retinaface import RetinaFace
        self.detector = RetinaFace

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            if self.method == 'haar':
                return self._detect_haar(frame)
            elif self.method == 'mtcnn':
                return self._detect_mtcnn(frame)
            elif self.method == 'retinaface':
                return self._detect_retinaface(frame)
        except:
            return []

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _detect_mtcnn(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)

        faces = []
        for det in detections:
            if det['confidence'] >= self.min_confidence:
                x, y, w, h = det['box']
                faces.append((max(0, x), max(0, y), abs(w), abs(h)))

        return faces

    def _detect_retinaface(self, frame):
        try:
            detections = self.detector.detect_faces(frame)
        except:
            return []

        faces = []
        if isinstance(detections, dict):
            for det in detections.values():
                if det['score'] >= self.min_confidence:
                    x1, y1, x2, y2 = det['facial_area']
                    w = x2 - x1
                    h = y2 - y1
                    faces.append((x1, y1, w, h))

        return faces

    def crop_faces(self, frame, padding=0.2):
        faces = self.detect(frame)

        if not faces:
            return []

        cropped = []
        h_img, w_img = frame.shape[:2]

        for (x, y, w, h) in faces:

            if w < 40 or h < 40:
                continue

            pad_w = int(w * padding)
            pad_h = int(h * padding)

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            cropped.append(face_crop)

        return cropped
