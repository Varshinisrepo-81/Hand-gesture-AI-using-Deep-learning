"""
TensorFlow/Keras Deep Learning Model Architecture for Hand Gesture Recognition.
A multi-layer feedforward deep neural network with BatchNormalization and Dropout
optimized for real-time inference latency and high generalization accuracy.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Optional

from src.preprocessor import GESTURE_CLASSES

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "gesture_model.keras")


def build_gesture_model(
    input_dim: int = 63,
    num_classes: int = len(GESTURE_CLASSES),
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Builds and compiles a deep neural network for hand gesture classification.

    Args:
        input_dim: Number of normalized input features (21 joints * 3 = 63).
        num_classes: Number of distinct gesture categories (default: 32).
        learning_rate: Initial learning rate for the Adam optimizer.

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=(input_dim,), name="landmark_input")

    # Layer 1
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.25, name="dropout_1")(x)

    # Layer 2
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.20, name="dropout_2")(x)

    # Layer 3
    x = layers.Dense(64, activation="relu", name="dense_3")(x)
    x = layers.Dropout(0.15, name="dropout_3")(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation="softmax", name="gesture_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="HandGestureNet")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_gesture_model(model: keras.Model, filepath: str = DEFAULT_MODEL_PATH) -> str:
    """Saves model in Keras native format (.keras)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    return filepath


def load_gesture_model(filepath: str = DEFAULT_MODEL_PATH) -> Optional[keras.Model]:
    """Loads a pre-trained Keras model from disk. Returns None if file does not exist."""
    if not os.path.exists(filepath):
        return None
    return keras.models.load_model(filepath)
