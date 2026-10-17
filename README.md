<<<<<<< HEAD
# Hand Gesture Recognition using Deep Learning and Streamlit 🖐️🤖

An end-to-end Computer Vision and Deep Learning system for real-time hand gesture recognition. Powered by **Google MediaPipe**, a **TensorFlow/Keras Deep Neural Network**, and an interactive **Streamlit** web application with **SQLite** audit logging.

---

## 📌 Project Overview

This application recognizes **32 distinct hand gestures** in real-time through a live webcam feed or uploaded images:
- **Everyday & Conversational:** 🖐️ Open Palm, ✊ Fist, 👍 Thumbs Up, 👎 Thumbs Down, ✌️ Peace Sign, ☝️ Pointing Up, 👌 OK Sign, 🤘 Rock Sign, 🤙 Call Me, 🛑 Stop Sign.
- **Numbers (ASL & Counting):** 0️⃣ Number 0, 1️⃣ Number 1, 2️⃣ Number 2, 3️⃣ Number 3, 4️⃣ Number 4, 5️⃣ Number 5, 3️⃣ Three Fingers, 4️⃣ Four Fingers.
- **Special Poses & Signs:** 🔫 Gun Sign, 🤟 Love You Sign, 🖖 Spock Salute, 🤙 Pinky Up, 🤏 Pinch, 🤞 Crossed Fingers, 🦅 Claw Hand, 👉 Pointing Forward, 👈 Thumb Left, 👉 Thumb Right, ⬇️ Two Fingers Down, 🫡 Salute, 💍 Ring Extension, 🧲 C-Shape.

### Key Capabilities
- **Real-Time Webcam Vision:** 30 FPS video processing with MediaPipe skeletal joint detection and OpenCV acceleration (`CAP_DSHOW` on Windows).
- **Invariant Normalization:** Landmark preprocessing ensures **translation invariance** (relative to wrist) and **scale invariance** (normalized by maximum hand span). Hand distance and positioning do not degrade accuracy.
- **Genuine Deep Neural Network:** TensorFlow/Keras multi-layer perceptron (Dense 256 → BatchNorm → Dropout(0.25) → Dense 128 → BatchNorm → Dropout(0.20) → Dense 64 → Dropout(0.15) → Dense 32 Softmax) trained with categorical cross-entropy.
- **Comprehensive Evaluation:** Accuracy, Precision, Recall, Macro F1-Score, Confusion Matrix Heatmaps, and Training/Validation loss curves on unseen test data across all 32 classes.
- **SQLite Database Storage:** Automatically or manually logs prediction history, confidence scores, timestamps, and model training sessions.
- **CSV Data Export:** One-click export of prediction history for external auditing and data analysis.
- **Triple-Mode Recognition:** Supports Live OpenCV Webcam, In-Browser Photo Snap (`st.camera_input`), and Image File Upload (`st.file_uploader`).

---

## 🏗️ Project Architecture

```
Dl/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Beginner-friendly project documentation
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core source code modules
│   ├── __init__.py             # Package marker
│   ├── camera.py               # CameraManager: OpenCV capture & resource cleanup
│   ├── hand_detector.py        # HandDetector: MediaPipe Landmarker & HUD renderer
│   ├── preprocessor.py         # Feature normalizer (translation & scale invariant)
│   ├── model.py                # TensorFlow/Keras neural network architecture
│   ├── predictor.py            # GesturePredictor: Model inference & confidence filtering
│   ├── train.py                # Model training pipeline & unseen evaluation
│   ├── evaluate.py             # Performance metrics, confusion matrix & plot curves
│   └── ui.py                   # UI design system, custom CSS, badges & metric cards
│
├── data/                       # Dataset directory
│   ├── generate_dataset.py     # Kinematic 3D skeleton dataset generator
│   └── gestures_dataset.csv    # 2,500 labeled hand gesture samples (63 features + labels)
│
├── database/                   # Persistent storage
│   ├── __init__.py             # Package marker
│   ├── db_manager.py           # DatabaseManager: SQLite CRUD operations & CSV export
│   └── gestures.db             # SQLite database storing predictions & training records
│
├── models/                     # Saved model artifacts
│   ├── gesture_model.keras     # Trained TensorFlow/Keras deep learning model
│   ├── hand_landmarker.task    # Google MediaPipe 3D Hand Landmarker model asset
│   ├── label_encoder.json      # Class index to gesture name mapping
│   ├── test_cache.npz          # Cached unseen test data for instant evaluation
│   └── training_metrics.json   # Quantitative metrics & history logs from last training run
│
└── tests/                      # Automated unit and integration tests
    ├── __init__.py             # Test package marker
    ├── test_database.py        # Tests for SQLite database CRUD and CSV export
    ├── test_preprocessor.py    # Tests for translation & scale invariance normalization
    ├── test_predictor.py       # Tests for model inference and confidence filtering
    ├── test_training.py        # Tests for model architecture and metrics loading
    └── test_app_integration.py # Integration tests for UI, camera, and predictor
```

---

## ⚙️ Environment Setup & Installation

### 1. Prerequisites
- **Python:** Version 3.9, 3.10, 3.11, or 3.12 recommended.
- **Webcam:** Any built-in or USB webcam (device index `0`).

### 2. Clone or Navigate to the Project
```bash
cd c:\Varshini\Automation\Dl
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell / Command Prompt)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation & How to Obtain Data

### Built-in Dataset (`data/gestures_dataset.csv`)
The project comes with a pre-generated, calibrated dataset containing **2,500 samples** (500 samples per gesture across 5 classes).
Each sample contains **63 features** representing normalized (x, y, z) coordinates for MediaPipe's 21 3D hand joints:
- `x_0, y_0, z_0` (Wrist) through `x_20, y_20, z_20` (Pinky tip).

### Generating Fresh Data via CLI
You can generate or expand the dataset with 3D rotation, scaling, and sensor noise variations:
```bash
python data/generate_dataset.py --samples 500 --output data/gestures_dataset.csv
```

### Using Real-World Image Datasets (e.g., Kaggle, HaGRID)
If you have a folder of real hand gesture images (e.g. from Kaggle's *Hand Gesture Recognition Database* or *HaGRID*):
1. Run each image through `HandDetector.detect(img)`.
2. For each detected hand, call `normalize_landmarks(landmarks)` to extract the 63 normalized coordinates.
3. Save the coordinates along with the gesture name into `data/gestures_dataset.csv`.
Because the model operates on normalized skeletal coordinates rather than raw pixels, it generalizes across any camera, skin tone, hand size, and background lighting!

---

## ⚡ Training the Deep Learning Model

### Option A: From the Streamlit Dashboard (Recommended)
1. Open the application (`streamlit run app.py`).
2. Navigate to **⚡ Model Training** in the sidebar.
3. Adjust hyperparameters (Epochs, Batch Size, Learning Rate).
4. Click **🔥 Start Model Training** and watch real-time epoch progress and validation metrics!

### Option B: From the Command Line
```bash
python src/train.py --epochs 35 --batch_size 32 --lr 0.001
```
The training pipeline automatically:
1. Loads `data/gestures_dataset.csv` and performs a stratified 70% Train / 15% Validation / 15% Test split.
2. Compiles the TensorFlow/Keras neural network with `EarlyStopping` and `ReduceLROnPlateau`.
3. Evaluates the best model weights on **unseen test data**.
4. Saves `models/gesture_model.keras`, `models/training_metrics.json`, and records the session in SQLite `training_sessions`.

---

## 🚀 Running the Streamlit Application

Start the web dashboard by executing:
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

### App Sections:
1. **🏠 Home:** Architecture overview, system health indicators, and target gesture guide.
2. **📷 Live Recognition:**
   - **Live Webcam Stream:** Click **▶️ Start Camera** to begin 30 FPS video streaming with skeletal joint rendering, HUD badge, and live probability distribution. Click **⏹️ Stop Camera** to safely release camera hardware.
   - **Browser Photo Snap:** Single-shot camera capture via the browser.
   - **Upload Test Image:** Upload any photo containing a hand gesture to test predictions.
3. **⚡ Model Training:** Inspect dataset statistics, configure epochs, and trigger model retraining with live progress feedback.
4. **📊 Evaluation:** View Test Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Confusion Matrix Heatmap, and training loss/accuracy curves.
5. **📜 Prediction History:** Explore logged predictions, filter by gesture or confidence score, view frequency distributions, and download as CSV.

---

## 🧪 Running Automated Unit Tests

Run the full test suite with:
```bash
python -m unittest discover tests
```
The test suite covers:
- **`test_database.py`:** SQLite table initialization, prediction logging, querying, filtering, statistics, and CSV bytes generation.
- **`test_preprocessor.py`:** Translation invariance (wrist at origin), scale invariance, batch normalization, and feature dimension validation.
- **`test_predictor.py`:** Neural network loading, inference structure, 5-class gesture classification, and low-confidence fallback behavior.
- **`test_training.py`:** Model architecture shapes, unseen test evaluation, and metrics persistence.
- **`test_app_integration.py`:** End-to-end integration between camera, UI constants, predictor, and database logging.

---

## 🛠️ Troubleshooting & FAQs

### 1. Camera does not open or shows black screen
- Verify no other application (Zoom, Teams, Skype, or other browser tab) is using your webcam.
- Try switching the **Webcam Index** in the dropdown (try `0`, `1`, or `2`).
- On Windows, OpenCV uses `cv2.CAP_DSHOW` by default for fast hardware access.

### 2. Prediction shows "Uncertain / Low Confidence"
- The application includes a confidence threshold (default 60%). If your hand is partially out of frame or transitioning between gestures, it marks the frame as uncertain.
- Ensure your hand is clearly visible with good room lighting.

### 3. Missing MediaPipe model file
- The detector automatically downloads `models/hand_landmarker.task` on first run if missing. You can also manually place the official task asset in `models/`.

### 4. Database Permission Error on Windows
- SQLite connections are managed using safe Python context managers (`@contextmanager`), ensuring connections are closed and file locks released automatically.

---

## 📜 License & Acknowledgments
- **MediaPipe:** Google MediaPipe Hand Landmarker Task
- **TensorFlow/Keras:** Deep Learning framework
- **Streamlit:** Python Web Application framework
- **OpenCV:** Real-time Computer Vision Library
=======
# Hand-gesture-AI-using-Deep-learning
Real-time Hand Gesture Recognition using Deep Learning, OpenCV, MediaPipe, Python, and Streamlit.
