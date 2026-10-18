"""
Hand Gesture Recognition using Deep Learning and Streamlit.
Main application orchestrating:
- Live Webcam Inference with MediaPipe & OpenCV
- TensorFlow/Keras Neural Network Training Pipeline
- Model Evaluation & Performance Visualizations
- SQLite Prediction History & CSV Data Export
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ui import (
    inject_custom_css,
    render_brand_header,
    render_metric_card,
    render_gesture_cards,
    render_softmax_bars,
    GESTURE_ICONS,
    GESTURE_DESCRIPTIONS
)
from src.camera import CameraManager
from src.hand_detector import HandDetector
from src.predictor import GesturePredictor
from src.preprocessor import GESTURE_CLASSES, FEATURE_DIM
from src.train import train_gesture_pipeline, DATASET_PATH, MODEL_SAVE_PATH
from src.evaluate import (
    get_latest_metrics,
    evaluate_model_on_cache,
    plot_confusion_matrix,
    plot_training_curves,
    get_per_class_table
)
from data.generate_dataset import generate_synthetic_dataset
from data.process_images import IMAGES_DIR, process_image_directory, ensure_image_directories
from data.create_sample_images import generate_100_images_dataset
from database.db_manager import DatabaseManager


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hand Gesture Recognition AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Styling
inject_custom_css()


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "last_logged_time" not in st.session_state:
    st.session_state.last_logged_time = 0.0

if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()

if "predictor" not in st.session_state:
    st.session_state.predictor = GesturePredictor()

if "detector" not in st.session_state:
    try:
        st.session_state.detector = HandDetector()
    except Exception:
        st.session_state.detector = None


def _require_detector() -> None:
    """Stops the app if the MediaPipe detector is unavailable."""
    if st.session_state.get("detector") is None:
        st.error("⚠️ MediaPipe hand detector could not initialize. Please restart the app or check your environment.")
        st.stop()


# ---------------------------------------------------------
# Sidebar Navigation & System Diagnostics
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2 style="margin:0; font-weight:800; color:#58a6ff;">🖐️ Gesture AI</h2>
            <div style="color:#8b949e; font-size:0.85rem; margin-top:2px;">Deep Learning & Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📷 Live Recognition",
            "⚡ Model Training",
            "📊 Evaluation",
            "📜 Prediction History"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📋 System Health")

    # Model status
    is_model_ready = st.session_state.predictor.is_model_loaded()
    if is_model_ready:
        st.markdown('<span class="pill-badge pill-success">● Model: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-badge pill-danger">● Model: Missing</span>', unsafe_allow_html=True)

    # Dataset status
    has_dataset = os.path.exists(DATASET_PATH)
    if has_dataset:
        try:
            pd.read_csv(DATASET_PATH, nrows=1)
            st.markdown('<span class="pill-badge pill-success">● Dataset: Ready</span>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<span class="pill-badge pill-warning">● Dataset: Error</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-badge pill-warning">● Dataset: Missing</span>', unsafe_allow_html=True)

    # Database status
    st.markdown('<span class="pill-badge pill-info">● SQLite: Connected</span>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander(f"🖐️ Supported Gestures ({len(GESTURE_CLASSES)})", expanded=False):
        for g_name in GESTURE_CLASSES:
            icon = GESTURE_ICONS.get(g_name, "•")
            st.markdown(f"**{icon} {g_name}**")


# ---------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------
if page == "🏠 Home":
    render_brand_header(
        title="Hand Gesture Recognition",
        subtitle="Real-Time Computer Vision & Deep Learning Neural Network System",
        badge_text="TensorFlow & MediaPipe"
    )

    st.markdown("### 🎯 Target Gesture Classes")
    render_gesture_cards()

    st.markdown("---")

    col_arch, col_guide = st.columns([3, 2])

    with col_arch:
        st.markdown("### 🧠 End-to-End Deep Learning Architecture")
        st.markdown(
            """
            This application implements a complete industrial-grade machine learning pipeline:
            
            1. **Video Ingestion:** Live OpenCV frame acquisition with Windows `CAP_DSHOW` acceleration.
            2. **MediaPipe Landmark Extraction:** Extracts 21 3D skeletal landmarks (x, y, z) per hand in real-time.
            3. **Invariant Preprocessing:** Normalizes landmarks relative to the wrist (landmark 0) and scales by maximum hand span, guaranteeing **translation and scale invariance**.
            4. **Deep Neural Network Classifier:** Feedforward deep neural network (Dense 256 → BatchNorm → Dropout 0.25 → Dense 128 → BatchNorm → Dropout 0.20 → Dense 64 → Dropout 0.15 [...]
            5. **Low-Confidence Filtering:** Threshold-based rejection filters out ambiguous or transitional poses.
            6. **Telemetry & Database Logging:** Automatic or manual logging of predictions to SQLite for auditing and CSV export.
            """
        )

        st.info(
            "💡 **Beginner Note:** The model does NOT require retraining on every app launch. "
            "Pre-trained weights are safely saved to `models/gesture_model.keras` and loaded instantly."
        )

    with col_guide:
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown(
            """
            - **Step 1:** Verify the model in the sidebar. If missing, navigate to **⚡ Model Training** and click *Start Model Training*.
            - **Step 2:** Go to **📷 Live Recognition**, select your camera, and click **▶️ Start Camera**.
            - **Step 3:** Hold your hand clearly in front of the webcam. Observe the live skeletal overlay and confidence score!
            - **Step 4:** Check **📊 Evaluation** to inspect the confusion matrix, or **📜 Prediction History** to download logged predictions as CSV.
            """
        )

        # Overview stats
        try:
            stats = st.session_state.db.get_prediction_statistics()
            st.markdown("#### 📈 Database Quick Metrics")
            m1, m2 = st.columns(2)
            with m1:
                render_metric_card("Total Predictions", str(stats["total_predictions"]))
            with m2:
                render_metric_card("Avg Confidence", f"{stats['average_confidence'] * 100:.1f}%")
        except Exception:
            st.warning("Database is empty or not ready yet.")


# ---------------------------------------------------------
# PAGE 2: LIVE RECOGNITION
# ---------------------------------------------------------
elif page == "📷 Live Recognition":
    render_brand_header(
        title="Live Gesture Recognition",
        subtitle="Real-time webcam inference with skeletal landmark tracking and HUD telemetry",
        badge_text="30 FPS Real-time Vision"
    )

    if not st.session_state.predictor.is_model_loaded():
        st.error(
            "⚠️ **Model Not Found!** No trained model detected at `models/gesture_model.keras`. "
            "Please go to the **⚡ Model Training** tab and train the model first."
        )
        st.stop()

    _require_detector()

    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "🎥 Live Webcam Stream",
        "📸 Browser Photo Snap",
        "🖼️ Upload Test Image"
    ])

    # ---------------- TAB 1: LIVE WEBCAM ----------------
    with rec_tab1:
        # Detected cameras
        avail_cams = CameraManager.list_available_cameras()
        if not avail_cams:
            avail_cams = [0]

        # Top Controls Row
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 2.2, 1.8, 1.8])

        with ctrl_col1:
            camera_idx = st.selectbox("Webcam Device", options=avail_cams, index=0, format_func=lambda i: f"Camera #{i}")

        with ctrl_col2:
            preset_name = st.selectbox(
                "Video Resolution",
                options=list(CameraManager.PRESETS.keys()),
                index=0,
                help="HD (1280x720) provides high sharpness and responsiveness. Full HD (1080p) provides maximum fidelity."
            )

        with ctrl_col3:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.30,
                max_value=0.95,
                value=0.60,
                step=0.05,
                help="Predictions below this confidence level are flagged as Uncertain."
            )

        with ctrl_col4:
            draw_skeleton = st.checkbox("Draw Skeleton", value=True, help="Renders MediaPipe hand joints and bones.")
            auto_log = st.checkbox("Auto-log to SQLite", value=False, help="Automatically logs confident predictions every 2s.")

        # Camera Quality & Enhancement Expander
        with st.expander("✨ Camera Quality & Visual Enhancements", expanded=False):
            enh_col1, enh_col2, enh_col3 = st.columns(3)
            with enh_col1:
                enable_clahe = st.checkbox(
                    "Adaptive Lighting (CLAHE)",
                    value=True,
                    help="Equalizes harsh shadows, dark knuckles, and backlighting in LAB color space."
                )
            with enh_col2:
                enable_sharpen = st.checkbox(
                    "Crisp Edge Sharpening",
                    value=True,
                    help="Sharpens finger contours and joint boundaries using unsharp masking."
                )
            with enh_col3:
                enable_denoise = st.checkbox(
                    "Low-Light Denoising",
                    value=False,
                    help="Applies bilateral smoothing to eliminate sensor grain in dim environments."
                )

            adj_col1, adj_col2 = st.columns(2)
            with adj_col1:
                brightness_val = st.slider("Brightness Adjustment", min_value=-50, max_value=50, value=0, step=5)
            with adj_col2:
                contrast_val = st.slider("Contrast Adjustment", min_value=0.8, max_value=1.5, value=1.0, step=0.05)

        # Camera Action Buttons
        btn_col1, btn_col2, status_col = st.columns([1.5, 1.5, 4])
        with btn_col1:
            if not st.session_state.camera_running:
                if st.button("▶️ Start Camera", type="primary", use_container_width=True):
                    st.session_state.camera_running = True
                    st.rerun()
            else:
                st.button("🟢 Camera Running", disabled=True, use_container_width=True)

        with btn_col2:
            if st.session_state.camera_running:
                if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
                    st.session_state.camera_running = False
                    st.rerun()

        with status_col:
            target_res = CameraManager.PRESETS.get(preset_name, (1280, 720))
            st.caption(f"🎯 Target Profile: **{preset_name}** | Enhancements: **{'CLAHE ' if enable_clahe else ''}{'Sharpen ' if enable_sharpen else ''}{'Denoise' if enable_denoise else 'Standard'}** | {target_res[0]}x{target_res[1]}")

        st.markdown("---")

        # Layout: Video on Left (3 cols), Telemetry on Right (2 cols)
        stream_col, telem_col = st.columns([3, 2])

        video_placeholder = stream_col.empty()
        telem_placeholder = telem_col.empty()

        if not st.session_state.camera_running:
            video_placeholder.info(
                "📷 **Camera is currently stopped.**\n\n"
                "Click **▶️ Start Camera** above to begin live hand gesture recognition. "
                "Ensure your webcam is connected and not opened in another application."
            )
            with telem_placeholder.container():
                st.markdown(
                    """
                    <div class="telemetry-tile">
                        <div style="font-size:3.5rem;">🔍</div>
                        <div class="telemetry-gesture-title" style="color:#8b949e;">Awaiting Stream</div>
                        <div class="telemetry-conf-label">Start the webcam to view real-time predictions</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            # Active Webcam Stream Loop with HD and Quality Enhancements
            req_w, req_h = CameraManager.PRESETS.get(preset_name, (1280, 720))
            cam = CameraManager(device_index=camera_idx, width=req_w, height=req_h)
            cam.set_enhancements(
                enable_clahe=enable_clahe,
                enable_sharpening=enable_sharpen,
                enable_denoising=enable_denoise,
                brightness=brightness_val,
                contrast=contrast_val
            )
            success, msg = cam.start()

            if not success:
                st.error(f"❌ {msg}")
                st.session_state.camera_running = False
            else:
                prev_time = time.time()
                fps = 0.0

                try:
                    while st.session_state.camera_running:
                        ret, frame = cam.read_frame()
                        if not ret or frame is None:
                            video_placeholder.warning("⚠️ Failed to grab frame from webcam. Stream stopping.")
                            break

                        # FPS calculation
                        curr_time = time.time()
                        dt = curr_time - prev_time
                        if dt > 0:
                            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                        prev_time = curr_time

                        # Detection
                        landmarks, handedness = st.session_state.detector.detect(frame)
                        annotated_frame = frame.copy()

                        gesture_res = None
                        if landmarks:
                            gesture_res = st.session_state.predictor.predict(
                                landmarks,
                                confidence_threshold=conf_threshold
                            )

                            if draw_skeleton:
                                annotated_frame = st.session_state.detector.draw_landmarks(
                                    annotated_frame,
                                    landmarks,
                                    handedness
                                )

                            annotated_frame = st.session_state.detector.draw_hud_overlay(
                                annotated_frame,
                                gesture=gesture_res["gesture"],
                                confidence=gesture_res["confidence"],
                                is_low_conf=gesture_res["is_low_confidence"],
                                fps=fps
                            )

                            if auto_log and not gesture_res["is_low_confidence"]:
                                if curr_time - st.session_state.last_logged_time > 2.0:
                                    st.session_state.db.log_prediction(
                                        gesture=gesture_res["raw_gesture"],
                                        confidence=gesture_res["confidence"],
                                        landmarks_count=21,
                                        notes=f"Auto-logged ({handedness})"
                                    )
                                    st.session_state.last_logged_time = curr_time
                        else:
                            annotated_frame = st.session_state.detector.draw_hud_overlay(
                                annotated_frame,
                                gesture="No Hand Detected",
                                confidence=0.0,
                                is_low_conf=False,
                                fps=fps
                            )

                        frame_rgb = frame[:, :, ::-1] if not draw_skeleton else annotated_frame[:, :, ::-1]
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        with telem_placeholder.container():
                            if gesture_res and landmarks:
                                disp_name = gesture_res["gesture"]
                                conf = gesture_res["confidence"]
                                is_low = gesture_res["is_low_confidence"]
                                icon = GESTURE_ICONS.get(gesture_res["raw_gesture"], "✋")
                                color = "#f59e0b" if is_low else "#10b981"

                                st.markdown(
                                    f"""
                                    <div class="telemetry-tile">
                                        <div style="font-size:3.5rem;">{icon}</div>
                                        <div class="telemetry-gesture-title" style="color:{color};">{disp_name}</div>
                                        <div class="telemetry-conf-label">Confidence: <b>{conf * 100:.1f}%</b> | Hand: <b>{handedness}</b></div>
                                        <div style="font-size:0.8rem; color:#8b949e; margin-top:6px;">Quality: <span style="color:#58a6ff; font-weight:600;">{cam.get_resolution_string()}</span> @ <span style="color:#3f8cff; font-weight:600;">{fps:.1f} FPS</span></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                render_softmax_bars(gesture_res["probabilities"])
                            else:
                                st.markdown(
                                    f"""
                                    <div class="telemetry-tile">
                                        <div style="font-size:3.5rem;">🔍</div>
                                        <div class="telemetry-gesture-title" style="color:#8b949e;">No Hand Detected</div>
                                        <div class="telemetry-conf-label">Position hand clearly within the camera frame</div>
                                        <div style="font-size:0.8rem; color:#8b949e; margin-top:6px;">Stream: <span style="color:#58a6ff;">{cam.get_resolution_string()}</span> @ <span style="color:#3f8cff;">{fps:.1f} FPS</span></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        time.sleep(0.015)

                finally:
                    cam.stop()
                    st.session_state.camera_running = False

    # ---------------- TAB 2: BROWSER PHOTO SNAP ----------------
    with rec_tab2:
        st.markdown("#### 📸 In-Browser Camera Capture")
        st.caption("Capture a quick snapshot using your web browser's camera with adaptive image enhancement.")

        snap_enh = st.checkbox("✨ Enhance Snapshot Clarity & Lighting", value=True, key="snap_enh_opt")

        snap_photo = st.camera_input("Take a hand photo")
        if snap_photo is not None:
            _require_detector()
            image_pil = Image.open(snap_photo)
            img_np = np.array(image_pil)
            img_bgr = img_np[:, :, ::-1]

            if snap_enh:
                temp_cam = CameraManager()
                img_bgr = temp_cam.enhance_frame(img_bgr)

            col_img, col_res = st.columns([3, 2])

            with col_img:
                landmarks, handed = st.session_state.detector.detect(img_bgr)
                if landmarks:
                    annotated = st.session_state.detector.draw_landmarks(img_bgr, landmarks, handed)
                    pred = st.session_state.predictor.predict(landmarks)
                    annotated = st.session_state.detector.draw_hud_overlay(
                        annotated,
                        gesture=pred["gesture"],
                        confidence=pred["confidence"],
                        is_low_conf=pred["is_low_confidence"]
                    )
                    st.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                else:
                    st.image(img_bgr[:, :, ::-1], channels="RGB", use_container_width=True)
                    st.warning("No hand landmarks detected in this photo. Please try again with clear lighting.")

            with col_res:
                if landmarks and pred:
                    st.success(f"**Predicted Gesture:** {pred['gesture']}")
                    st.metric("Confidence Score", f"{pred['confidence'] * 100:.1f}%")
                    render_softmax_bars(pred["probabilities"])

                    if st.button("💾 Save Snapshot Prediction to Database"):
                        st.session_state.db.log_prediction(
                            gesture=pred["raw_gesture"],
                            confidence=pred["confidence"],
                            landmarks_count=21,
                            notes=f"Browser photo snap ({handed})"
                        )
                        st.toast("Prediction successfully recorded in database!", icon="✅")

    # ---------------- TAB 3: IMAGE UPLOAD ----------------
    with rec_tab3:
        st.markdown("#### 🖼️ Test With Uploaded Image")
        st.caption("Upload a JPG or PNG image containing a visible hand to test the prediction pipeline.")

        up_enh = st.checkbox("✨ Enhance Image Contrast & Sharpness", value=True, key="up_enh_opt")

        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            _require_detector()
            image_pil = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image_pil)
            img_bgr = img_np[:, :, ::-1]

            if up_enh:
                temp_cam = CameraManager()
                img_bgr = temp_cam.enhance_frame(img_bgr)

            col_u_img, col_u_res = st.columns([3, 2])

            with col_u_img:
                landmarks, handed = st.session_state.detector.detect(img_bgr)
                if landmarks:
                    annotated = st.session_state.detector.draw_landmarks(img_bgr, landmarks, handed)
                    pred = st.session_state.predictor.predict(landmarks)
                    annotated = st.session_state.detector.draw_hud_overlay(
                        annotated,
                        gesture=pred["gesture"],
                        confidence=pred["confidence"],
                        is_low_conf=pred["is_low_confidence"]
                    )
                    st.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                else:
                    st.image(img_np, channels="RGB", use_container_width=True)
                    st.warning("No hand landmarks detected in this image.")

            with col_u_res:
                if landmarks and pred:
                    st.success(f"**Recognized Gesture:** {pred['gesture']}")
                    st.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")
                    render_softmax_bars(pred["probabilities"])

                    if st.button("💾 Log Upload Prediction to History"):
                        st.session_state.db.log_prediction(
                            gesture=pred["raw_gesture"],
                            confidence=pred["confidence"],
                            landmarks_count=21,
                            notes=f"Uploaded image: {uploaded_file.name}"
                        )
                        st.toast("Prediction saved!", icon="✅")


# ---------------------------------------------------------
# PAGE 3: MODEL TRAINING
# ---------------------------------------------------------
elif page == "⚡ Model Training":
    render_brand_header(
        title="Model Training Studio",
        subtitle="Train a genuine TensorFlow/Keras neural network with live progress callbacks",
        badge_text="TensorFlow Keras MLP"
    )

    t_col1, t_col2 = st.columns([3, 2])

    with t_col1:
        st.markdown("### 📊 Dataset Preparation")

        ds_tab1, ds_tab2 = st.tabs(["🖼️ 100 Image Dataset", "⚙️ Kinematic Generator"])

        with ds_tab1:
            st.markdown("#### 📁 Image Folders (`data/images/`)")
            img_counts = {}
            for g in GESTURE_CLASSES:
                p = os.path.join(IMAGES_DIR, g)
                os.makedirs(p, exist_ok=True)
                img_counts[g] = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

            total_imgs = sum(img_counts.values())

            # Grid of counts
            st.caption(f"Total collected images: **{total_imgs} images** across gesture image folders.")
            num_cols = 4
            for row_idx in range(0, len(GESTURE_CLASSES), num_cols):
                cols = st.columns(num_cols)
                for col_idx, g in enumerate(GESTURE_CLASSES[row_idx:row_idx + num_cols]):
                    with cols[col_idx]:
                        count = img_counts.get(g, 0)
                        st.metric(g, f"{count} imgs")

            # Gallery expander
            with st.expander("🔍 Preview Image Dataset Samples"):
                preview_classes = [g for g in GESTURE_CLASSES if img_counts.get(g, 0) > 0]
                if not preview_classes:
                    st.info("No sample images found on disk.")
                else:
                    g_cols = st.columns(min(len(preview_classes), 5))
                    for idx, g in enumerate(preview_classes[:5]):
                        with g_cols[idx]:
                            p = os.path.join(IMAGES_DIR, g)
                            files = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                            if files:
                                sample_img = Image.open(os.path.join(p, files[0]))
                                st.image(sample_img, caption=f"{g} (#{len(files)})", use_container_width=True)

            img_action_col1, img_action_col2 = st.columns(2)
            with img_action_col1:
                if st.button("📸 Process Images to Training CSV", type="secondary", use_container_width=True):
                    with st.spinner("Extracting 21 3D landmarks with MediaPipe from image folders..."):
                        proc_stats = process_image_directory(images_dir=IMAGES_DIR, output_csv=DATASET_PATH)
                    st.success(
                        f"Extracted {proc_stats['successful_detections']}/{proc_stats['total_images']} "
                        f"samples to {DATASET_PATH}!"
                    )
                    st.rerun()

            with img_action_col2:
                if st.button("🔄 Regenerate Sample Images", use_container_width=True):
                    with st.spinner("Generating labeled hand gesture images..."):
                        generate_100_images_dataset(images_per_class=20)
                    st.success("Generated hand gesture images successfully!")
                    st.rerun()

        with ds_tab2:
            st.markdown("#### ⚙️ Kinematic 3D Generator")
            st.write("Generates synthetic 3D hand skeletons with anatomical noise and Euler rotations.")
            samples_per_class = st.slider("Samples per gesture class", min_value=100, max_value=1000, value=500, step=50)

            if st.button("🔨 Generate Kinematic Dataset"):
                with st.spinner("Generating calibrated landmark dataset..."):
                    generate_synthetic_dataset(samples_per_class=samples_per_class, output_path=DATASET_PATH)
                st.success(f"Generated {samples_per_class * len(GESTURE_CLASSES):,} samples successfully!")
                st.rerun()

        if os.path.exists(DATASET_PATH):
            df_ds = pd.read_csv(DATASET_PATH)
            st.info(f"📋 **Current Active CSV:** `{len(df_ds):,}` samples ready for training.")

    with t_col2:
        st.markdown("### 🎛️ Training Hyperparameters")
        epochs = st.slider("Epochs", min_value=10, max_value=100, value=35, step=5)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        learning_rate = st.selectbox("Learning Rate", options=[0.001, 0.0005, 0.0001], index=0)

        st.markdown("#### 📐 Neural Network Architecture")
        st.code(
            f"""Input (63 features)
├── Dense(256, ReLU) + BatchNorm + Dropout(0.25)
├── Dense(128, ReLU) + BatchNorm + Dropout(0.20)
├── Dense(64, ReLU) + Dropout(0.15)
└── Dense({len(GESTURE_CLASSES)}, Softmax) -> Gesture Probabilities
Optimizer: Adam | Loss: Sparse Categorical Crossentropy""",
            language="text"
        )

    st.markdown("---")
    st.markdown("### 🚀 Execute Model Training")

    train_btn = st.button("🔥 Start Model Training", type="primary", use_container_width=True)

    if train_btn:
        if not os.path.exists(DATASET_PATH):
            st.error("Please generate or provide the dataset first!")
        else:
            progress_bar = st.progress(0)
            epoch_status = st.empty()
            loss_status = st.empty()

            def on_progress(epoch, total_epochs, logs):
                pct = epoch / total_epochs
                progress_bar.progress(pct)
                loss_val = logs.get("loss", 0.0)
                val_loss_val = logs.get("val_loss", 0.0)
                acc_val = logs.get("accuracy", 0.0)
                val_acc_val = logs.get("val_accuracy", 0.0)

                epoch_status.markdown(f"**Epoch {epoch} of {total_epochs}** ({int(pct * 100)}% complete)")
                loss_status.markdown(
                    f"• Train Loss: `{loss_val:.4f}` | Val Loss: `{val_loss_val:.4f}`\n"
                    f"• Train Acc: `{acc_val * 100:.2f}%` | Val Acc: `{val_acc_val * 100:.2f}%`"
                )

            with st.spinner("Training TensorFlow/Keras neural network..."):
                metrics = train_gesture_pipeline(
                    dataset_path=DATASET_PATH,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=on_progress
                )

            progress_bar.progress(1.0)
            st.success(f"🎉 **Training Complete!** Model evaluated on **{metrics['test_samples']} unseen test samples**.")

            st.session_state.predictor.load_model_and_labels()

            res1, res2, res3, res4 = st.columns(4)
            with res1:
                render_metric_card("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            with res2:
                render_metric_card("Macro Precision", f"{metrics['precision_macro'] * 100:.2f}%")
            with res3:
                render_metric_card("Macro Recall", f"{metrics['recall_macro'] * 100:.2f}%")
            with res4:
                render_metric_card("Macro F1-Score", f"{metrics['f1_macro'] * 100:.2f}%")


# ---------------------------------------------------------
# PAGE 4: EVALUATION
# ---------------------------------------------------------
elif page == "📊 Evaluation":
    render_brand_header(
        title="Model Evaluation & Analytics",
        subtitle="Quantitative metrics, confusion matrix, and training performance on unseen data",
        badge_text="Scikit-Learn & Keras Metrics"
    )

    metrics = get_latest_metrics()

    if not metrics:
        st.warning("⚠️ No evaluation metrics found. Please train the model in the **⚡ Model Training** tab first.")
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_metric_card("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%", f"{metrics.get('test_samples', 0)} samples")
        with k2:
            render_metric_card("Precision (Macro)", f"{metrics['precision_macro'] * 100:.2f}%")
        with k3:
            render_metric_card("Recall (Macro)", f"{metrics['recall_macro'] * 100:.2f}%")
        with k4:
            render_metric_card("F1-Score (Macro)", f"{metrics['f1_macro'] * 100:.2f}%")
        with k5:
            render_metric_card("Test Loss", f"{metrics['test_loss']:.4f}")

        st.markdown("---")

        v_col1, v_col2 = st.columns([1, 1.3])

        with v_col1:
            st.markdown("### 🧩 Confusion Matrix (Unseen Test Data)")
            cm_fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics.get("classes", GESTURE_CLASSES))
            st.pyplot(cm_fig, use_container_width=True)

        with v_col2:
            st.markdown("### 📈 Training & Validation Curves")
            if "history" in metrics:
                hist_fig = plot_training_curves(metrics["history"])
                st.pyplot(hist_fig, use_container_width=True)
            else:
                st.info("Training history curves not available.")

        st.markdown("---")

        st.markdown("### 📋 Per-Class Performance Breakdown")
        if "classification_report" in metrics:
            df_class = get_per_class_table(metrics["classification_report"])
            st.dataframe(df_class, use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Re-evaluate Model on Cached Test Set"):
            with st.spinner("Running evaluation on cached test set..."):
                try:
                    new_eval = evaluate_model_on_cache()
                    st.success(f"Re-evaluation complete! Test Accuracy: {new_eval['accuracy'] * 100:.2f}%")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during re-evaluation: {e}")


# ---------------------------------------------------------
# PAGE 5: PREDICTION HISTORY
# ---------------------------------------------------------
elif page == "📜 Prediction History":
    render_brand_header(
        title="Prediction History & Audit Logs",
        subtitle="SQLite database records of live inferences, confidence scores, and training history",
        badge_text="SQLite Storage"
    )

    db: DatabaseManager = st.session_state.db
    try:
        stats = db.get_prediction_statistics()
    except Exception:
        stats = {
            "total_predictions": 0,
            "average_confidence": 0.0,
            "max_confidence": 0.0,
            "most_frequent_gesture": "None",
            "gesture_breakdown": {}
        }

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        render_metric_card("Total Inferences", str(stats["total_predictions"]))
    with h2:
        render_metric_card("Average Confidence", f"{stats['average_confidence'] * 100:.1f}%")
    with h3:
        render_metric_card("Highest Confidence", f"{stats['max_confidence'] * 100:.1f}%")
    with h4:
        render_metric_card("Top Recognized Gesture", stats["most_frequent_gesture"])

    st.markdown("---")

    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 2])

    with f_col1:
        gesture_options = ["All"] + GESTURE_CLASSES
        selected_gesture = st.selectbox("Filter by Gesture", options=gesture_options, index=0)

    with f_col2:
        min_conf = st.slider("Minimum Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    with f_col3:
        record_limit = st.selectbox("Max Records", options=[50, 100, 250, 500, 1000], index=1)

    with f_col4:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        csv_bytes = db.export_predictions_csv_bytes()
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name=f"hand_gesture_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    df_preds = db.get_predictions_df(
        limit=record_limit,
        gesture_filter=selected_gesture if selected_gesture != "All" else None,
        min_confidence=min_conf
    )

    st.markdown(f"### 📋 Logged Predictions ({len(df_preds)} records)")

    if len(df_preds) == 0:
        st.info("No prediction records match the current filter criteria.")
    else:
        display_df = df_preds.copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda v: f"{float(v) * 100:.2f}%")
        st.dataframe(display_df, use_container_width=True)

    if stats["gesture_breakdown"]:
        st.markdown("### 📊 Gesture Frequency Distribution")
        df_chart = pd.DataFrame(
            list(stats["gesture_breakdown"].items()),
            columns=["Gesture", "Count"]
        ).set_index("Gesture")
        st.bar_chart(df_chart)

    st.markdown("---")
    with st.expander("🗑️ Database Maintenance"):
        st.write("Clear all prediction history from the SQLite database.")
        confirm_clear = st.checkbox("I confirm I want to erase all prediction records.")
        if st.button("Delete All Predictions", type="secondary", disabled=not confirm_clear):
            db.clear_predictions()
            st.success("All prediction records have been cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 🕒 Model Training Sessions Log")
    df_train_logs = db.get_training_sessions_df(limit=25)
    if len(df_train_logs) > 0:
        train_disp = df_train_logs.copy()
        train_disp["test_accuracy"] = train_disp["test_accuracy"].apply(lambda v: f"{float(v) * 100:.2f}%")
        st.dataframe(train_disp, use_container_width=True)
    else:
        st.info("No model training sessions logged yet.")

""",
"path":"app.py",
"repo":"Varshinisrepo-81/Hand-gesture-AI-using-Deep-learning"
},{"content":"""
Gesture Prediction Engine for Hand Gesture Recognition.
Connects landmark preprocessor with the trained TensorFlow/Keras deep learning model.
Handles class mapping, probability distribution, and low-confidence filtering.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import numpy as np
from typing import Dict, Any, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import normalize_landmarks, extract_landmarks_from_mediapipe, GESTURE_CLASSES
from src.model import load_gesture_model, DEFAULT_MODEL_PATH

DEFAULT_LABEL_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.json")


class GesturePredictor:
    """Performs real-time gesture inference given hand landmarks."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, label_path: str = DEFAULT_LABEL_PATH):
        self.model_path = model_path
        self.label_path = label_path
        self.model = None
        self.label_map = {i: name for i, name in enumerate(GESTURE_CLASSES)}
        self.load_model_and_labels()

    def is_model_loaded(self) -> bool:
        """Returns True if the deep learning model is successfully loaded."""
        return self.model is not None

    def load_model_and_labels(self) -> bool:
        """Loads or reloads the trained Keras model and label mapping."""
        if os.path.exists(self.model_path):
            try:
                self.model = load_gesture_model(self.model_path)
            except Exception as e:
                print(f"[!] Error loading model: {e}")
                self.model = None

        if os.path.exists(self.label_path):
            try:
                with open(self.label_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                    self.label_map = {int(k): v for k, v in raw_map.items()}
            except Exception as e:
                print(f"[!] Error loading label mapping: {e}")

        return self.model is not None

    def predict(
        self,
        landmarks: Any,
        confidence_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Runs neural network inference on raw hand landmarks.

        Args:
            landmarks: MediaPipe landmark list or numpy coordinate array (21, 3).
            confidence_threshold: Threshold below which prediction is marked 'Uncertain'.

        Returns:
            Dictionary with prediction results, confidence score, and probabilities.
        """
        if self.model is None:
            return {
                "gesture": "Model Not Loaded",
                "raw_gesture": "None",
                "confidence": 0.0,
                "is_low_confidence": True,
                "probabilities": {},
                "normalized_features": None,
                "error": "No trained model found on disk. Please train the model first."
            }

        if isinstance(landmarks, (list, tuple)):
            if len(landmarks) == 0:
                return {
                    "gesture": "No Hand Detected",
                    "raw_gesture": "None",
                    "confidence": 0.0,
                    "is_low_confidence": True,
                    "probabilities": {},
                    "normalized_features": None,
                    "error": "No hand landmarks detected."
                }
            if hasattr(landmarks[0], "x"):
                raw_coords = extract_landmarks_from_mediapipe(landmarks)
            else:
                raw_coords = np.asarray(landmarks, dtype=np.float32)
        else:
            raw_coords = np.asarray(landmarks, dtype=np.float32)

        try:
            norm_features = normalize_landmarks(raw_coords)
        except Exception as e:
            return {
                "gesture": "Invalid Landmark Input",
                "raw_gesture": "None",
                "confidence": 0.0,
                "is_low_confidence": True,
                "probabilities": {},
                "normalized_features": None,
                "error": str(e)
            }

        input_tensor = np.expand_dims(norm_features, axis=0)
        probs = self.model.predict(input_tensor, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_gesture = self.label_map.get(pred_idx, f"Class {pred_idx}")

        is_low_conf = confidence < confidence_threshold
        if is_low_conf:
            gesture_display = "Uncertain / Low Confidence"
        else:
            gesture_display = raw_gesture

        prob_dict = {
            self.label_map.get(i, f"Class {i}"): round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        return {
            "gesture": gesture_display,
            "raw_gesture": raw_gesture,
            "confidence": round(confidence, 4),
            "is_low_confidence": is_low_conf,
            "probabilities": prob_dict,
            "normalized_features": norm_features,
            "error": None
        }

""",
"path":"src/predictor.py",
"repo":"Varshinisrepo-81/Hand-gesture-AI-using-Deep-learning"
}]} mbalactic in this? Need careful: JSON content includes triple quotes with 
