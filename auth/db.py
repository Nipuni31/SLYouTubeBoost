"""SQLite user store for JWT authentication."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parents[1]
INSTANCE_DIR = BASE_DIR / "instance"
DB_PATH = INSTANCE_DIR / "users.db"


def get_conn() -> sqlite3.Connection:
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    ph = generate_password_hash(password, method="pbkdf2:sha256")
    now = datetime.now(timezone.utc).isoformat()
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, ph, now),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "Username already taken."
    return True, "OK"


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def verify_user(username: str, password: str) -> bool:
    user = get_user_by_username(username)
    if user is None:
        return False
    return check_password_hash(user["password_hash"], password)


def ensure_default_admin() -> None:
    """Create default admin from env if no users exist."""
    import os

    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if n > 0:
        return
    user = os.environ.get("DEFAULT_ADMIN_USER", "admin")
    pw = os.environ.get("DEFAULT_ADMIN_PASSWORD", "changeme123")
    ok, msg = create_user(user, pw)
    if ok:
        print(f"Created default user '{user}' (change DEFAULT_ADMIN_PASSWORD in production).")
    else:
        print(f"Could not create default admin: {msg}")
