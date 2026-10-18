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
        from src.hand_detector import HandDetector
        st.session_state.detector = HandDetector()
    except Exception as e:
        st.session_state.detector = None
        # Log error quietly without crashing the UI thread
        print(f"MediaPipe initialization warning: {e}")
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

    # Detector status
    if st.session_state.detector is not None:
        st.markdown('<span class="pill-badge pill-success">● MediaPipe: Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-badge pill-danger">● MediaPipe: Error</span>', unsafe_allow_html=True)

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

    if st.session_state.detector is None:
        st.error(
            "⚠️ **MediaPipe HandDetector Failed to Initialize!** "
            "Please check your package dependencies (`mediapipe`, `opencv-python`) in your environment."
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
                    annotated = st.session_state.detector.draw_landmarks(img_bgr, landmarks, handed)
                    pred = st.session_state.predictor.predict(landmarks)
                    annotated = st.session_state.detector.draw_hud_overlay(
                        annotated, gesture=pred["gesture"], confidence=pred["confidence"], is_low_conf=pred["is_low_confidence"]
                    )
                    st.image(annotated[:, :, ::-1], channels="RGB", use_container_width=True)
                else:
                    st.image(img_np, channels="RGB", use_container_width=True)
                    st.warning("No hand landmarks detected.")

            with col_u_res:
                if landmarks and pred:
                    st.success(f"**Recognized:** {pred['gesture']}")
                    st.metric("Confidence", f"{pred['confidence'] * 100:.1f}%")
                    render_softmax_bars(pred["probabilities"])
                    if st.button("💾 Log Upload Prediction"):
                        st.session_state.db.log_prediction(
                            gesture=pred["raw_gesture"], confidence=pred["confidence"], landmarks_count=21, notes=f"Uploaded: {uploaded_file.name}"
                        )
                        st.toast("Saved!", icon="✅")


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
            st.caption(f"Total collected images: **{total_imgs} images**")
            
            num_cols = 4
            for row_idx in range(0, len(GESTURE_CLASSES), num_cols):
                cols = st.columns(num_cols)
                for col_idx, g in enumerate(GESTURE_CLASSES[row_idx:row_idx + num_cols]):
                    with cols[col_idx]:
                        count = img_counts.get(g, 0)
                        st.metric(g, f"{count} imgs")

            img_action_col1, img_action_col2 = st.columns(2)
            with img_action_col1:
                if st.button("📸 Process Images to Training CSV", type="secondary", use_container_width=True):
                    with st.spinner("Extracting landmarks..."):
                        proc_stats = process_image_directory(images_dir=IMAGES_DIR, output_csv=DATASET_PATH)
                    st.success(f"Extracted {proc_stats['successful_detections']}/{proc_stats['total_images']} samples!")
                    st.rerun()

            with img_action_col2:
                if st.button("🔄 Regenerate Sample Images", use_container_width=True):
                    with st.spinner("Generating sample images..."):
                        generate_100_images_dataset(images_per_class=20)
                    st.success("Sample images generated successfully!")
                    st.rerun()

        with ds_tab2:
            st.markdown("#### ⚙️ Kinematic 3D Generator")
            samples_per_class = st.slider("Samples per gesture class", min_value=100, max_value=1000, value=500, step=50)
            if st.button("🔨 Generate Kinematic Dataset"):
                with st.spinner("Generating dataset..."):
                    generate_synthetic_dataset(samples_per_class=samples_per_class, output_path=DATASET_PATH)
                st.success("Generated samples successfully!")
                st.rerun()

        if os.path.exists(DATASET_PATH):
            df_ds = pd.read_csv(DATASET_PATH)
            st.info(f"📋 **Current Active CSV:** `{len(df_ds):,}` samples ready.")

    with t_col2:
        st.markdown("### 🎛️ Training Hyperparameters")
        epochs = st.slider("Epochs", min_value=10, max_value=100, value=35, step=5)
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
        learning_rate = st.selectbox("Learning Rate", options=[0.001, 0.0005, 0.0001], index=0)

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
                epoch_status.markdown(f"**Epoch {epoch} of {total_epochs}**")
                loss_status.markdown(f"• Train Loss: `{logs.get('loss', 0):.4f}` | Val Acc: `{logs.get('val_accuracy', 0) * 100:.2f}%`")

            with st.spinner("Training TensorFlow/Keras neural network..."):
                metrics = train_gesture_pipeline(
                    dataset_path=DATASET_PATH,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=on_progress
                )

            progress_bar.progress(1.0)
            st.success("🎉 **Training Complete!**")

            st.session_state.predictor.load_model_and_labels()

            res1, res2, res3, res4 = st.columns(4)
            with res1:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0) * 100:.2f}%")
            with res2:
                st.metric("Test Loss", f"{metrics.get('test_loss', 0):.4f}")
            with res3:
                st.metric("Test Samples", str(metrics.get('test_samples', 0)))
            with res4:
                st.metric("Classes", str(len(GESTURE_CLASSES)))


# ---------------------------------------------------------
# PAGE 4: EVALUATION
# ---------------------------------------------------------
elif page == "📊 Evaluation":
    render_brand_header(
        title="Model Evaluation & Analytics",
        subtitle="Deep inspection of confusion matrices, per-class metrics, and training loss curves",
        badge_text="Performance Metrics"
    )

    if not st.session_state.predictor.is_model_loaded():
        st.error("⚠️ Model not found. Please train the model first in the **⚡ Model Training** tab.")
        st.stop()

    eval_tab1, eval_tab2, eval_tab3 = st.tabs([
        "📈 Confusion Matrix",
        "📉 Training Curves",
        "📋 Per-Class Report"
    ])

    with eval_tab1:
        st.markdown("#### 🔍 Confusion Matrix on Test Split")
        try:
            fig_cm = plot_confusion_matrix()
            if fig_cm:
                st.pyplot(fig_cm)
            else:
                st.info("Confusion matrix plot could not be generated.")
        except Exception as e:
            st.warning(f"Could not render confusion matrix: {e}")

    with eval_tab2:
        st.markdown("#### 📈 Training & Validation Curves")
        try:
            fig_curves = plot_training_curves()
            if fig_curves:
                st.pyplot(fig_curves)
            else:
                st.info("Training history not found.")
        except Exception as e:
            st.warning(f"Could not render training curves: {e}")

    with eval_tab3:
        st.markdown("#### 📋 Per-Class Performance Breakdown")
        try:
            per_class_df = get_per_class_table()
            if per_class_df is not None and not per_class_df.empty:
                st.dataframe(per_class_df, use_container_width=True)
            else:
                st.info("Per-class performance table unavailable.")
        except Exception as e:
            st.warning(f"Could not load per-class report: {e}")


# ---------------------------------------------------------
# PAGE 5: PREDICTION HISTORY
# ---------------------------------------------------------
elif page == "📜 Prediction History":
    render_brand_header(
        title="Prediction History & Telemetry",
        subtitle="Inspect logged predictions stored in SQLite and export telemetry reports as CSV",
        badge_text="SQLite & CSV Export"
    )

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("### 📋 Recent Model Inferences")
    with col_h2:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            st.session_state.db.clear_history()
            st.toast("Prediction history cleared!", icon="🗑️")
            st.rerun()

    history_df = st.session_state.db.get_prediction_history(limit=100)

    if history_df.empty:
        st.info(
            "📭 **No predictions logged yet.**\n\n"
            "Use **📷 Live Recognition** with 'Auto-log to SQLite' enabled, or log manual snapshots/uploads."
        )
    else:
        st.dataframe(history_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Export Options")
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full History as CSV",
            data=csv_data,
            file_name=f"gesture_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
