"""
Model Training Pipeline for Hand Gesture Recognition.
Handles:
- Dataset loading and validation
- Stratified train/validation/test splitting (70% / 15% / 15%)
- Genuine TensorFlow/Keras neural network training with callbacks
- Unseen test set evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Saving model, metrics, and SQLite training record
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tensorflow import keras

import sys
# Ensure paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "gestures_dataset.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.keras")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "training_metrics.json")
TEST_CACHE_PATH = os.path.join(PROJECT_ROOT, "models", "test_cache.npz")

from src.preprocessor import GESTURE_CLASSES, FEATURE_DIM
from src.model import build_gesture_model, save_gesture_model
from database.db_manager import DatabaseManager


class StreamlitProgressCallback(keras.callbacks.Callback):
    """Callback to update Streamlit or external progress bars epoch-by-epoch."""

    def __init__(self, total_epochs: int, callback_fn: Optional[Callable[[int, int, Dict[str, float]], None]] = None):
        super().__init__()
        self.total_epochs = total_epochs
        self.callback_fn = callback_fn

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        if self.callback_fn and logs:
            self.callback_fn(epoch + 1, self.total_epochs, logs)


def load_and_split_data(
    dataset_path: str = DATASET_PATH,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Loads dataset from CSV and splits into stratified Train, Validation, and Test sets.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please generate or provide it first.")

    df = pd.read_csv(dataset_path)

    # Feature columns (x_0 .. z_20)
    feature_cols = [c for c in df.columns if c.startswith(("x_", "y_", "z_"))]
    if len(feature_cols) != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} feature columns, found {len(feature_cols)}.")

    X = df[feature_cols].values.astype(np.float32)

    # Label column
    if "label" in df.columns:
        y = df["label"].values.astype(np.int32)
    elif "gesture" in df.columns:
        gesture_to_id = {g: i for i, g in enumerate(GESTURE_CLASSES)}
        y = df["gesture"].map(gesture_to_id).values.astype(np.int32)
    else:
        raise ValueError("Dataset must contain 'label' or 'gesture' column.")

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_gesture_pipeline(
    dataset_path: str = DATASET_PATH,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None
) -> Dict[str, Any]:
    """
    Executes the complete end-to-end training and evaluation pipeline.
    """
    # 1. Load and split data
    data = load_and_split_data(dataset_path=dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"[*] Train set: {len(X_train)} samples")
    print(f"[*] Val set:   {len(X_val)} samples")
    print(f"[*] Test set:  {len(X_test)} samples (Unseen evaluation)")

    # 2. Build Model
    model = build_gesture_model(
        input_dim=FEATURE_DIM,
        num_classes=len(GESTURE_CLASSES),
        learning_rate=learning_rate
    )

    # 3. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    if progress_callback:
        callbacks.append(StreamlitProgressCallback(total_epochs=epochs, callback_fn=progress_callback))

    # 4. Train Model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate on Unseen Test Set
    test_eval = model.evaluate(X_test, y_test, verbose=0)
    test_loss = float(test_eval[0])
    test_acc = float(test_eval[1])

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate Scikit-Learn Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, target_names=GESTURE_CLASSES, output_dict=True, zero_division=0)

    # 6. Save Artifacts
    save_gesture_model(model, filepath=MODEL_SAVE_PATH)

    # Save label encoder mapping
    label_map = {str(i): name for i, name in enumerate(GESTURE_CLASSES)}
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    # Save metrics JSON
    metrics_summary = {
        "accuracy": round(accuracy, 4),
        "test_loss": round(test_loss, 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weight), 4),
        "recall_weighted": round(float(r_weight), 4),
        "f1_weighted": round(float(f1_weight), 4),
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "classes": GESTURE_CLASSES,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epochs_trained": len(history.history["loss"]),
        "history": {
            "loss": [round(float(v), 4) for v in history.history["loss"]],
            "val_loss": [round(float(v), 4) for v in history.history.get("val_loss", [])],
            "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
            "val_accuracy": [round(float(v), 4) for v in history.history.get("val_accuracy", [])],
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # Save test cache for offline evaluation page reload
    np.savez_compressed(TEST_CACHE_PATH, X_test=X_test, y_test=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs)

    # 7. Log to SQLite
    db = DatabaseManager()
    db.log_training_session(
        epochs=len(history.history["loss"]),
        train_samples=len(X_train),
        test_samples=len(X_test),
        test_accuracy=accuracy,
        test_loss=test_loss,
        model_path=MODEL_SAVE_PATH
    )

    print("\n" + "="*50)
    print(" TRAINING SUMMARY ON UNSEEN TEST DATA")
    print("="*50)
    print(f" Test Accuracy:  {accuracy * 100:.2f}%")
    print(f" Macro Precision:{p_macro * 100:.2f}%")
    print(f" Macro Recall:   {r_macro * 100:.2f}%")
    print(f" Macro F1-Score: {f1_macro * 100:.2f}%")
    print("="*50)

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hand Gesture Recognition Model")
    parser.add_argument("--epochs", type=int, default=35, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_gesture_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
