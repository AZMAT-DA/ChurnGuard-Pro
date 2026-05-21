# database.py — SQLite database handler for user management

import sqlite3
import bcrypt
import os
from datetime import datetime

# Database file location — stored in your project folder
DB_PATH = 'users.db'


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def initialize_database():
    """
    Create all tables if they do not exist yet.
    This runs every time the app starts — safe to call multiple times.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ---- Users table ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    DEFAULT 'user',
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP,
            last_login    TEXT
        )
    ''')

    # ---- Predictions table ----
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id            INTEGER NOT NULL,
            username           TEXT    NOT NULL,
            date               TEXT,
            time               TEXT,
            tenure             TEXT,
            monthly_charges    TEXT,
            contract           TEXT,
            internet           TEXT,
            churn_probability  TEXT,
            prediction         TEXT,
            risk_level         TEXT,
            created_at         TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()

    # ---- Create default admin account if it does not exist ----
    existing = cursor.execute(
        "SELECT id FROM users WHERE username = 'admin'"
    ).fetchone()

    if not existing:
        hashed = bcrypt.hashpw('1234'.encode(), bcrypt.gensalt())
        cursor.execute('''
            INSERT INTO users (full_name, email, username, password_hash, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('Administrator', 'admin@churnpro.com',
              'admin', hashed.decode(), 'admin'))
        conn.commit()
        print("Default admin account created: admin / 1234")

    conn.close()


def register_user(full_name: str, email: str,
                  username: str, password: str) -> dict:
    """
    Register a new user.
    Returns {'success': True} or {'success': False, 'error': 'message'}
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Check if username already exists
        existing_user = cursor.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing_user:
            return {'success': False,
                    'error': 'Username already taken. Choose a different one.'}

        # Check if email already exists
        existing_email = cursor.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing_email:
            return {'success': False,
                    'error': 'Email already registered. Try logging in.'}

        # Hash the password — never store plain text
        password_hash = bcrypt.hashpw(
            password.encode(), bcrypt.gensalt()
        ).decode()

        # Insert new user
        cursor.execute('''
            INSERT INTO users (full_name, email, username, password_hash, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (full_name, email, username, password_hash, 'user'))

        conn.commit()
        return {'success': True}

    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def login_user(username: str, password: str) -> dict:
    """
    Verify username and password.
    Returns user data dict if successful, or error message.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Find user by username
        user = cursor.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if not user:
            return {'success': False,
                    'error': 'Username not found. Please register first.'}

        # Check password against stored hash
        password_correct = bcrypt.checkpw(
            password.encode(),
            user['password_hash'].encode()
        )

        if not password_correct:
            return {'success': False,
                    'error': 'Wrong password. Please try again.'}

        # Update last login time
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().strftime('%Y-%m-%d %H:%M'), user['id'])
        )
        conn.commit()

        return {
            'success'  : True,
            'user_id'  : user['id'],
            'username' : user['username'],
            'full_name': user['full_name'],
            'email'    : user['email'],
            'role'     : user['role']
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def save_prediction(user_id: int, username: str,
                    customer_data: dict, prob: float,
                    pred: int, risk_level: str):
    """Save a prediction to the database for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO predictions
            (user_id, username, date, time, tenure, monthly_charges,
             contract, internet, churn_probability, prediction, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            username,
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%H:%M'),
            str(customer_data.get('tenure', '')),
            str(customer_data.get('monthly_charges', '')),
            str(customer_data.get('contract', '')),
            str(customer_data.get('internet', '')),
            f"{prob:.2%}",
            'CHURN' if pred == 1 else 'STAY',
            risk_level
        ))
        conn.commit()
    except Exception as e:
        print(f"Error saving prediction: {e}")
    finally:
        conn.close()


def get_user_predictions(user_id: int):
    """Get all predictions for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_predictions():
    """Get all predictions — for admin only."""
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_users():
    """Get all registered users — for admin only."""
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT id, full_name, email, username, role, created_at, last_login FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_user_count():
    """Return total number of registered users."""
    conn = get_connection()
    cursor = conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return count


def delete_user(user_id: int):
    """Delete a user and their predictions — admin only."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()