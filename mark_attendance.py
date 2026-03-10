import sqlite3
from datetime import datetime

DB_FILE = "attendance.db"


def mark_attendance(name):

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if attendance already marked today
    cursor.execute("""
        SELECT 1 FROM attendance
        WHERE name = ? AND date = ?
    """, (name, date))

    already_marked = cursor.fetchone()

    if already_marked is None:

        cursor.execute("""
            INSERT INTO attendance (name, date, time)
            VALUES (?, ?, ?)
        """, (name, date, time))

        conn.commit()

        print(f"✓ Attendance marked for {name} at {time}")

    else:
        print(f"⚠ {name} already marked today")

    conn.close()