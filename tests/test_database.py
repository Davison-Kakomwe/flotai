"""
tests/test_database.py
Unit and integration tests for database.py.

Uses a temporary, isolated database for every test - never touches
the real data/flotai.db. This is critical: tests must never risk
corrupting real application data.
"""

import sqlite3
import pytest
import sys
from pathlib import Path

# Allow imports from the project root (one level up from tests/)
sys.path.append(str(Path(__file__).parent.parent))

import database


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Creates a brand-new, temporary SQLite database file for each test.
    `tmp_path` is a special pytest fixture that gives us a unique
    temporary folder that gets cleaned up automatically after the test.

    `monkeypatch` temporarily replaces database.DB_PATH with our temp
    file, so every function in database.py points at the test DB
    instead of the real one - without changing any of our source code.
    """
    test_db_path = tmp_path / "test_flotai.db"
    monkeypatch.setattr(database, "DB_PATH", test_db_path)

    database.init_db()
    database.seed_default_data()

    yield test_db_path  # test runs here

    # No manual cleanup needed - tmp_path deletes itself automatically


def test_init_db_creates_tables(temp_db):
    """
    Confirms all expected tables exist after init_db() runs.
    """
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = {row[0] for row in cursor.fetchall()}
    conn.close()

    expected_tables = {"plants", "users", "flotation_readings", "froth_features", "lab_assays"}
    assert expected_tables.issubset(table_names)


def test_seed_default_data_creates_plant_and_user(temp_db):
    """
    Confirms seed_default_data() actually inserts a plant and a user,
    which our foreign keys depend on.
    """
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM plants")
    plant_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]

    conn.close()

    assert plant_count >= 1
    assert user_count >= 1


def test_seed_default_data_is_idempotent(temp_db):
    """
    'Idempotent' means running it multiple times has the same effect
    as running it once - it shouldn't create duplicate plants/users
    every time the app restarts.
    """
    database.seed_default_data()
    database.seed_default_data()  # run it again

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM plants")
    plant_count = cursor.fetchone()[0]
    conn.close()

    assert plant_count == 1  # should still be exactly 1, not 2 or 3


def test_save_reading_returns_valid_id(temp_db):
    """
    Confirms save_reading() successfully inserts a row and returns
    a usable ID.
    """
    reading_id = database.save_reading(
        image_path="videos/test.mp4",
        predicted_recovery=85.5,
        predicted_grade=24.0,
        confidence_score=0.9,
    )

    assert isinstance(reading_id, int)
    assert reading_id > 0


def test_save_reading_rejects_invalid_recovery(temp_db):
    """
    Our CHECK constraint should reject recovery values outside 0-100.
    This proves our database-level validation (Step 6) actually works,
    not just our Python code's assumptions.
    """
    with pytest.raises(sqlite3.IntegrityError):
        database.save_reading(
            image_path="videos/test.mp4",
            predicted_recovery=150.0,  # invalid - over 100
            predicted_grade=24.0,
            confidence_score=0.9,
        )


def test_save_froth_features_links_to_reading(temp_db):
    """
    Confirms froth features are correctly linked to their parent
    reading via reading_id - our one-to-one relationship from Step 6.
    """
    reading_id = database.save_reading(
        image_path="videos/test.mp4",
        predicted_recovery=80.0,
        predicted_grade=22.0,
        confidence_score=0.85,
    )

    database.save_froth_features(
        reading_id=reading_id,
        avg_bubble_size=500.0,
        color_hue_avg=60.0,
        texture_score=300.0,
        froth_speed=1.5,
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT reading_id, avg_bubble_size FROM froth_features WHERE reading_id = ?", (reading_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == reading_id
    assert row[1] == 500.0


def test_get_recent_readings_returns_newest_first(temp_db):
    """
    Confirms readings come back ordered by timestamp descending.
    """
    database.save_reading("videos/1.mp4", 70.0, 20.0, 0.8)
    database.save_reading("videos/2.mp4", 75.0, 21.0, 0.8)

    rows = database.get_recent_readings(limit=10)

    assert len(rows) == 2
    # rows[0] should be the most recently inserted (id=2)
    assert rows[0][0] == 2