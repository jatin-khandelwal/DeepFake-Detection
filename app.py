import streamlit as st
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

from inference import DeepfakePredictor
from deepfake_model.preprocessing.face_detector import FaceDetector

st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.title("🎭 Deepfake Detector")

# ==== MODEL PATHS ====
checkpoint_paths = {
    "efficientnet": "deepfake_model/checkpoints/efficientnet/best_model.pth",
    "xception": "deepfake_model/checkpoints/xception/best_model.pth",
    "vit": "deepfake_model/checkpoints/vit/best_model.pth",
}

@st.cache_resource
def load_models():
    return DeepfakePredictor(checkpoint_paths)

@st.cache_resource
def load_detector():
    return FaceDetector(method="mtcnn")  # better detector

predictor = load_models()
detector = load_detector()

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ==== THRESHOLDS ====
FAKE_THRESHOLD = 0.4
UNCERTAIN_LOW = 0.2
UNCERTAIN_HIGH = 0.3


# ==== IMAGE ====
def process_image(file):

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    h, w = image.shape[:2]

    # Show ORIGINAL image centered
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption=f"Original Image ({w} x {h})",
            use_container_width=False
        )

    faces = detector.crop_faces(image)

    if len(faces) == 0:
        st.error("No face detected")
        return

    all_probs = []

    st.subheader("Model Predictions")

    for i, face in enumerate(faces):

        results = predictor.predict_all(face)

        st.write(f"### Face {i+1}")

        st.write(f"EfficientNet: {results['efficientnet']:.2f}")
        st.write(f"Xception: {results['xception']:.2f}")
        st.write(f"ViT: {results['vit']:.2f}")

        st.bar_chart({
            "EfficientNet": results["efficientnet"],
            "Xception": results["xception"],
            "ViT": results["vit"]
        })

        prob = results["ensemble"]
        all_probs.append(prob)

        # SMALL FACE ONLY
        st.image(face, width=120, caption=f"Prob: {prob:.2f}")

        st.markdown("---")

    final_score = float(np.median(all_probs))

    st.subheader("FINAL DECISION")
    st.write(f"Final Score: {final_score:.2f}")
    st.progress(final_score)

    if final_score > FAKE_THRESHOLD:
        st.error("FAKE / AI GENERATED")
    elif UNCERTAIN_LOW < final_score <= UNCERTAIN_HIGH:
        st.warning("UNCERTAIN")
    else:
        st.success("REAL IMAGE")


# ==== VIDEO ====
def process_video(file):

    video_bytes = file.read()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)

    # ORIGINAL VIDEO DISPLAY
    st.video(video_bytes)

    video = cv2.VideoCapture(tfile.name)

    frame_count = 0
    probs = []

    progress = st.progress(0)

    MAX_FRAMES = 120
    FRAME_SKIP = 10

    st.subheader("Model Predictions per Frame")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        faces = detector.crop_faces(frame)
        faces = faces[:2]

        for face in faces:

            results = predictor.predict_all(face)

            st.write(f"### Frame {frame_count}")

            st.write(f"EfficientNet: {results['efficientnet']:.2f}")
            st.write(f"Xception: {results['xception']:.2f}")
            st.write(f"ViT: {results['vit']:.2f}")

            st.bar_chart({
                "EfficientNet": results["efficientnet"],
                "Xception": results["xception"],
                "ViT": results["vit"]
            })

            prob = results["ensemble"]
            probs.append(prob)

            st.image(face, width=100, caption=f"Prob: {prob:.2f}")

            st.markdown("---")

        frame_count += 1
        progress.progress(min(frame_count / MAX_FRAMES, 1.0))

        if frame_count > MAX_FRAMES:
            break

    video.release()

    if len(probs) == 0:
        st.error("No face detected")
        return

    # GRAPH
    st.subheader("Frame-wise Fake Probability")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(probs, linewidth=2)
    ax.axhline(y=FAKE_THRESHOLD, linestyle="--")

    ax.set_title("Fake Probability per Frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    st.pyplot(fig)

    # TOP-K MEDIAN
    top_k = sorted(probs, reverse=True)[:10]
    final_score = float(np.median(top_k))

    st.subheader("FINAL DECISION")
    st.write(f"Final Score: {final_score:.2f}")
    st.progress(final_score)

    if final_score > FAKE_THRESHOLD:
        st.error("FAKE / AI VIDEO")
    elif UNCERTAIN_LOW < final_score <= UNCERTAIN_HIGH:
        st.warning("UNCERTAIN VIDEO")
    else:
        st.success("REAL VIDEO")


# ==== MAIN ====
if uploaded_file:

    if "image" in uploaded_file.type:
        process_image(uploaded_file)

    elif "video" in uploaded_file.type:
        process_video(uploaded_file)

    else:
        st.error("Unsupported file type")
