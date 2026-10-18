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
    except Exception as e:
        st.session_state.detector = None


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
            df_info = pd.read_csv(DATASET_PATH, nrows=1)
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
            4. **Deep Neural Network Classifier:** Feedforward deep neural network (Dense 256 → BatchNorm → Dropout 0.25 → Dense 128 → BatchNorm → Dropout 0.20 → Dense 64 → Dropout 0.15 → Dense 32 Softmax).
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
        stats = st.session_state.db.get_prediction_statistics()
        st.markdown("#### 📈 Database Quick Metrics")
        m1, m2 = st.columns(2)
        with m1:
            render_metric_card("Total Predictions", str(stats["total_predictions"]))
        with m2:
            render_metric_card("Avg Confidence", f"{stats['average_confidence'] * 100:.1f}%")


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

    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "🎥 Live Webcam Stream",
        "📸 Browser Photo Snap",
        "🖼️ Upload Test Image"
    ])

    # ---------------- TAB 1: LIVE WEBCAM ----------------
    with rec_tab1:
        avail_cams = CameraManager.list_available_cameras()
        if not avail_cams:
            avail_cams = [0]

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

        with st.expander("✨ Camera Quality & Visual Enhancements", expanded=False):
            enh_col1, enh_col2, enh_col3 = st.columns(3)
            with enh_col1:
                enable_clahe = st.checkbox("Adaptive Lighting (CLAHE)", value=True)
            with enh_col2:
                enable_sharpen = st.checkbox("Crisp Edge Sharpening", value=True)
            with enh_col3:
                enable_denoise = st.checkbox("Low-Light Denoising", value=False)

            adj_col1, adj_col2 = st.columns(2)
            with adj_col1:
                brightness_val = st.slider("Brightness Adjustment", min_value=-50, max_value=50, value=0, step=5)
            with adj_col2:
                contrast_val = st.slider("Contrast Adjustment", min_value=0.8, max_value=1.5, value=1.0, step=0.05)

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

        st.markdown("---")

        stream_col, telem_col = st.columns([3, 2])
        video_placeholder = stream_col.empty()
        telem_placeholder = telem_col.empty()

        if not st.session_state.camera_running:
            video_placeholder.info(
                "📷 **Camera is currently stopped.**\n\n"
                "Click **▶️ Start Camera** above to begin live hand gesture recognition."
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

                        curr_time = time.time()
                        dt = curr_time - prev_time
                        if dt > 0:
                            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                        prev_time = curr_time

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
                                    annotated_frame, landmarks, handedness
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
                                        <div style="font-size:0.8rem; color:#8b949e; margin-top:6px;">Quality: <span style="color:#58a6ff; font-weight:600;">{cam.get_resolution_string()}</span> @ <span style="color:#3fb950; font-weight:600;">{fps:.0f} FPS</span></div>
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
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        time.sleep(0.015)
                finally:
                    cam.stop()

    # ---------------- TAB 2: BROWSER PHOTO SNAP ----------------
    with rec_tab2:
        st.markdown("#### 📸 In-Browser Camera Capture")
        snap_enh = st.checkbox("✨ Enhance Snapshot Clarity & Lighting", value=True, key="snap_enh_opt")
        snap_photo = st.camera_input("Take a hand photo")
        
        if snap_photo is not None:
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
                        annotated, gesture=pred["gesture"], confidence=pred["confidence"], is_low_conf=pred["is_low_confidence"]
                    )
                    st.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                else:
                    st.image(img_bgr[:, :, ::-1], channels="RGB", use_container_width=True)
                    st.warning("No hand landmarks detected.")

            with col_res:
                if landmarks and pred:
                    st.success(f"**Predicted:** {pred['gesture']}")
                    st.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")
                    render_softmax_bars(pred["probabilities"])
                    if st.button("💾 Save Snapshot Prediction"):
                        st.session_state.db.log_prediction(
                            gesture=pred["raw_gesture"], confidence=pred["confidence"], landmarks_count=21, notes=f"Snapshot ({handed})"
                        )
                        st.toast("Prediction recorded!", icon="✅")

    # ---------------- TAB 3: IMAGE UPLOAD ----------------
    with rec_tab3:
        st.markdown("#### 🖼️ Test With Uploaded Image")
        up_enh = st.checkbox("✨ Enhance Image Contrast & Sharpness", value=True, key="up_enh_opt")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
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
                    annotated = st.session_state.detector.draw_landmarks(img_b
