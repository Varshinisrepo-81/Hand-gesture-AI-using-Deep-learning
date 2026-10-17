"""
Model Evaluation and Metrics Visualization Module.
Calculates and plots:
- Accuracy, Precision, Recall, F1-Score (Macro and Weighted)
- Confusion Matrix with styled Heatmap
- Per-class Performance Metrics Breakdown
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessor import GESTURE_CLASSES
from src.model import load_gesture_model

METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "training_metrics.json")
TEST_CACHE_PATH = os.path.join(PROJECT_ROOT, "models", "test_cache.npz")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.keras")


def get_latest_metrics() -> Optional[Dict[str, Any]]:
    """Loads saved metrics dictionary from disk if available."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def evaluate_model_on_cache() -> Dict[str, Any]:
    """
    Evaluates the current trained model on the cached unseen test set.
    """
    if not os.path.exists(TEST_CACHE_PATH):
        raise FileNotFoundError(f"Test cache not found at {TEST_CACHE_PATH}. Run training first.")

    cache = np.load(TEST_CACHE_PATH)
    X_test = cache["X_test"]
    y_test = cache["y_test"]

    model = load_gesture_model(MODEL_PATH)
    if model is None:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    test_eval = model.evaluate(X_test, y_test, verbose=0)
    test_loss = float(test_eval[0])
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_dict = classification_report(y_test, y_pred, target_names=GESTURE_CLASSES, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "test_loss": round(test_loss, 4),
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weight), 4),
        "recall_weighted": round(float(r_weight), 4),
        "f1_weighted": round(float(f1_weight), 4),
        "confusion_matrix": cm,
        "classification_report": clf_dict,
        "test_samples": len(y_test),
        "classes": GESTURE_CLASSES
    }


def plot_confusion_matrix(cm_data: list, class_names: list = GESTURE_CLASSES) -> plt.Figure:
    """
    Creates an aesthetically styled Confusion Matrix heatmap figure.
    Dynamically scales dimensions and typography for 30+ classes.
    """
    cm_array = np.array(cm_data)
    num_classes = len(class_names)

    # Dynamic figure sizing and typography scaling for 30+ classes
    if num_classes > 20:
        fig_size = (15, 12)
        annot_size = 7
        tick_size = 8
        show_annot = True
        line_w = 0.5
    elif num_classes > 10:
        fig_size = (10, 8)
        annot_size = 9
        tick_size = 9
        show_annot = True
        line_w = 1.0
    else:
        fig_size = (7, 5.5)
        annot_size = 12
        tick_size = 10
        show_annot = True
        line_w = 1.5

    fig, ax = plt.subplots(figsize=fig_size, facecolor="#0e1117")
    ax.set_facecolor("#161b22")

    # Seaborn heatmap with vibrant custom color map
    sns.heatmap(
        cm_array,
        annot=show_annot,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=line_w,
        linecolor="#0e1117",
        annot_kws={"size": annot_size, "weight": "bold", "color": "#ffffff" if num_classes > 20 else "#000000"}
    )

    ax.set_title(f"Confusion Matrix on Unseen Test Data ({num_classes} Classes)", fontsize=14, weight="bold", pad=15, color="#ffffff")
    ax.set_xlabel("Predicted Gesture", fontsize=12, labelpad=10, weight="bold", color="#58a6ff")
    ax.set_ylabel("Actual Gesture", fontsize=12, labelpad=10, weight="bold", color="#58a6ff")

    ax.tick_params(colors="#c9d1d9", labelsize=tick_size)
    plt.xticks(rotation=45, ha="right", fontsize=tick_size)
    plt.yticks(rotation=0, fontsize=tick_size)
    plt.tight_layout()

    return fig


def plot_training_curves(history_dict: Dict[str, list]) -> plt.Figure:
    """
    Generates training & validation curves for Loss and Accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0e1117")

    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    # Accuracy Plot
    ax1.set_facecolor("#161b22")
    ax1.plot(epochs, history_dict.get("accuracy", []), label="Train Accuracy", color="#2ea043", linewidth=2.5)
    if "val_accuracy" in history_dict and len(history_dict["val_accuracy"]) > 0:
        ax1.plot(epochs, history_dict["val_accuracy"], label="Val Accuracy", color="#58a6ff", linestyle="--", linewidth=2.2)
    ax1.set_title("Model Accuracy over Epochs", fontsize=12, weight="bold", color="#ffffff")
    ax1.set_xlabel("Epoch", color="#c9d1d9")
    ax1.set_ylabel("Accuracy", color="#c9d1d9")
    ax1.set_ylim([0.8, 1.02])
    ax1.grid(True, linestyle=":", alpha=0.3, color="#8b949e")
    ax1.tick_params(colors="#c9d1d9")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

    # Loss Plot
    ax2.set_facecolor("#161b22")
    ax2.plot(epochs, history_dict.get("loss", []), label="Train Loss", color="#f85149", linewidth=2.5)
    if "val_loss" in history_dict and len(history_dict["val_loss"]) > 0:
        ax2.plot(epochs, history_dict["val_loss"], label="Val Loss", color="#d29922", linestyle="--", linewidth=2.2)
    ax2.set_title("Model Loss over Epochs", fontsize=12, weight="bold", color="#ffffff")
    ax2.set_xlabel("Epoch", color="#c9d1d9")
    ax2.set_ylabel("Loss", color="#c9d1d9")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#8b949e")
    ax2.tick_params(colors="#c9d1d9")
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9")

    plt.tight_layout()
    return fig


def get_per_class_table(clf_report: Dict[str, Any]) -> pd.DataFrame:
    """Extracts per-class precision, recall, f1, and support into a styled DataFrame."""
    rows = []
    for cls in GESTURE_CLASSES:
        if cls in clf_report:
            item = clf_report[cls]
            rows.append({
                "Gesture Class": cls,
                "Precision": f"{item['precision'] * 100:.2f}%",
                "Recall": f"{item['recall'] * 100:.2f}%",
                "F1-Score": f"{item['f1-score'] * 100:.2f}%",
                "Support (Samples)": int(item["support"])
            })
    return pd.DataFrame(rows)
