"""
database.py
Handles all database setup and queries for FlotAI.
Keeping all DB logic in one file means the rest of our app
never needs to know *how* data is stored - just what functions to call.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "flotai.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensures the 'data' folder exists, on any machine/server


def get_connection():
    """
    Opens and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """
    Creates all our tables if they don't already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            role TEXT CHECK (role IN ('operator','engineer','admin')) NOT NULL,
            plant_id INTEGER REFERENCES plants(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flotation_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            image_path TEXT NOT NULL,
            predicted_recovery REAL NOT NULL CHECK (predicted_recovery BETWEEN 0 AND 100),
            predicted_grade REAL NOT NULL CHECK (predicted_grade BETWEEN 0 AND 100),
            confidence_score REAL,
            plant_id INTEGER REFERENCES plants(id),
            created_by INTEGER REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_readings_timestamp
        ON flotation_readings(timestamp)
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS froth_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER REFERENCES flotation_readings(id),
            avg_bubble_size REAL,
            color_hue_avg REAL,
            texture_score REAL,
            froth_speed REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lab_assays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_id INTEGER REFERENCES flotation_readings(id),
            actual_recovery REAL,
            actual_grade REAL,
            assay_timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


def seed_default_data():
    """
    Ensures at least one plant and one user exist, so foreign key
    references (plant_id=1, created_by=1) used elsewhere always
    point to real rows. Safe to call every time the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM plants")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO plants (name, location) VALUES (?, ?)",
            ("Demo Concentrator Plant", "Copperbelt, Zambia")
        )

    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, role, plant_id) VALUES (?, ?, ?)",
            ("demo_operator", "operator", 1)
        )

    conn.commit()
    conn.close()


def save_reading(image_path, predicted_recovery, predicted_grade, confidence_score, plant_id=1, created_by=1):
    """
    Inserts a new flotation reading into the database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO flotation_readings
        (image_path, predicted_recovery, predicted_grade, confidence_score, plant_id, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (image_path, predicted_recovery, predicted_grade, confidence_score, plant_id, created_by))

    reading_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return reading_id


def save_froth_features(reading_id, avg_bubble_size, color_hue_avg, texture_score, froth_speed):
    """
    Saves the extracted froth features linked to a specific reading.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO froth_features
        (reading_id, avg_bubble_size, color_hue_avg, texture_score, froth_speed)
        VALUES (?, ?, ?, ?, ?)
    """, (reading_id, avg_bubble_size, color_hue_avg, texture_score, froth_speed))

    conn.commit()
    conn.close()


def get_recent_readings(limit=50):
    """
    Fetches the most recent flotation readings, newest first.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, timestamp, predicted_recovery, predicted_grade, confidence_score
        FROM flotation_readings
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    init_db()
    seed_default_data()
    print("Database initialized successfully.")