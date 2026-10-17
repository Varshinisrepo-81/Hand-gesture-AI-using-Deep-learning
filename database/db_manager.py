"""
Database Manager for Hand Gesture Recognition System.
Handles SQLite storage and retrieval for prediction history and training session logs.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "gestures.db")


class DatabaseManager:
    """Manages SQLite operations for predictions and model training logs."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    @contextmanager
    def _get_connection(self):
        """Yields a SQLite connection with row factory enabled and guarantees closure."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initializes database tables if they do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Table for prediction logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    gesture TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    landmarks_count INTEGER DEFAULT 21,
                    notes TEXT DEFAULT ''
                )
            """)

            # Table for model training records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    epochs INTEGER NOT NULL,
                    train_samples INTEGER NOT NULL,
                    test_samples INTEGER NOT NULL,
                    test_accuracy REAL NOT NULL,
                    test_loss REAL NOT NULL,
                    model_path TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_prediction(
        self,
        gesture: str,
        confidence: float,
        landmarks_count: int = 21,
        notes: str = ""
    ) -> int:
        """
        Logs a single gesture prediction record.
        Returns the inserted row ID.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (timestamp, gesture, confidence, landmarks_count, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, gesture, round(float(confidence), 4), landmarks_count, notes),
            )
            conn.commit()
            return cursor.lastrowid

    def get_predictions_df(
        self,
        limit: int = 200,
        gesture_filter: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> pd.DataFrame:
        """Fetches prediction history as a pandas DataFrame with optional filters."""
        query = "SELECT id, timestamp, gesture, confidence, landmarks_count, notes FROM predictions WHERE confidence >= ?"
        params: List[Any] = [min_confidence]

        if gesture_filter and gesture_filter != "All":
            query += " AND gesture = ?"
            params.append(gesture_filter)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Calculates summary statistics across all prediction records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(confidence), MAX(confidence) FROM predictions")
            row = cursor.fetchone()
            total_count = row[0] if row else 0
            avg_conf = row[1] if (row and row[1] is not None) else 0.0
            max_conf = row[2] if (row and row[2] is not None) else 0.0

            # Frequency by gesture
            cursor.execute(
                "SELECT gesture, COUNT(*) as count FROM predictions GROUP BY gesture ORDER BY count DESC"
            )
            gesture_counts = {r["gesture"]: r["count"] for r in cursor.fetchall()}

            most_frequent = next(iter(gesture_counts.keys())) if gesture_counts else "None"

        return {
            "total_predictions": total_count,
            "average_confidence": round(float(avg_conf), 4),
            "max_confidence": round(float(max_conf), 4),
            "most_frequent_gesture": most_frequent,
            "gesture_breakdown": gesture_counts,
        }

    def clear_predictions(self) -> None:
        """Clears all records in the predictions table."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM predictions")
            conn.commit()

    def log_training_session(
        self,
        epochs: int,
        train_samples: int,
        test_samples: int,
        test_accuracy: float,
        test_loss: float,
        model_path: str
    ) -> int:
        """Logs a completed model training session."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO training_sessions (timestamp, epochs, train_samples, test_samples, test_accuracy, test_loss, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    int(epochs),
                    int(train_samples),
                    int(test_samples),
                    round(float(test_accuracy), 4),
                    round(float(test_loss), 4),
                    model_path,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_training_sessions_df(self, limit: int = 50) -> pd.DataFrame:
        """Fetches history of model training sessions."""
        query = "SELECT * FROM training_sessions ORDER BY id DESC LIMIT ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[limit])
        return df

    def export_predictions_csv_bytes(self) -> bytes:
        """Exports all predictions to CSV format encoded in UTF-8 bytes for Streamlit download."""
        df = self.get_predictions_df(limit=10000)
        return df.to_csv(index=False).encode("utf-8")
