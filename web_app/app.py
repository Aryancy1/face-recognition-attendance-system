from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from deepface import DeepFace
from numpy.linalg import norm
import base64
import cv2
import sqlite3
import math
from datetime import datetime

app = Flask(__name__)

# ---------------- SETTINGS ----------------

EMBEDDINGS_FILE = "../embeddings.pkl"
DB_FILE = "../attendance.db"

MODEL_NAME = "Facenet512"
THRESHOLD = 0.5

COLLEGE_LAT = 21.163367
COLLEGE_LON = 81.659399
ALLOWED_RADIUS_METERS = 20000


# ---------------- DATABASE INIT ----------------

def init_db():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------- LOAD EMBEDDINGS ----------------

def load_embeddings():

    with open(EMBEDDINGS_FILE, "rb") as f:
        db = pickle.load(f)

    for person in db:
        emb = np.array(db[person])
        db[person] = emb / norm(emb)

    return db


stored_embeddings = load_embeddings()


# ---------------- FACE DETECTOR ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ---------------- HAVERSINE DISTANCE ----------------

def haversine(lat1, lon1, lat2, lon2):

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------- DATABASE FUNCTIONS ----------------

def mark_attendance(name):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM attendance WHERE name=? AND date=?",
        (name, today)
    )

    if cursor.fetchone():
        conn.close()
        return False, "Already marked today"

    cursor.execute(
        "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
        (name, today, current_time)
    )

    conn.commit()
    conn.close()

    return True, current_time


# ---------------- FACE RECOGNITION ----------------

def recognize_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return "Unknown", 0

    (x, y, w, h) = faces[0]

    face_img = frame[y:y+h, x:x+w]

    if face_img.size == 0:
        return "Unknown", 0

    face_img = cv2.resize(face_img, (160, 160))

    result = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL_NAME,
        enforce_detection=False
    )

    embedding = np.array(result[0]["embedding"])
    embedding = embedding / norm(embedding)

    best_match = "Unknown"
    min_distance = float("inf")

    for person, stored_embedding in stored_embeddings.items():

        distance = 1 - np.dot(embedding, stored_embedding)

        if distance < min_distance:
            min_distance = distance
            best_match = person

    similarity = 1 - min_distance

    print("Best match:", best_match, "Similarity:", similarity)

    return best_match, similarity


# ---------------- ROUTES ----------------

@app.route("/")
def dashboard():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance")
    students = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = date('now')")
    today = cursor.fetchone()[0]

    conn.close()

    return render_template("dashboard.html", students=students, today=today)


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/verify", methods=["POST"])
def verify():

    try:

        data = request.json

        lat = float(data["latitude"])
        lon = float(data["longitude"])
        image_data = data["image"]

        image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # -------- LOCATION CHECK --------

        distance = haversine(lat, lon, COLLEGE_LAT, COLLEGE_LON)

        if distance > ALLOWED_RADIUS_METERS:

            return jsonify({
                "status": "Outside Campus",
                "distance_meters": round(distance, 2)
            })

        # -------- FACE RECOGNITION --------

        name, score = recognize_face(frame)

        if score < THRESHOLD:

            return jsonify({
                "status": "Face Not Recognized",
                "confidence": round(score * 100, 2)
            })

        # -------- ATTENDANCE --------

        success, message = mark_attendance(name)

        if not success:

            return jsonify({
                "status": message,
                "name": name,
                "confidence": round(score * 100, 2)
            })

        return jsonify({
            "status": "Attendance Marked",
            "name": name,
            "confidence": round(score * 100, 2),
            "time": message
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "status": "Error",
            "message": str(e)
        })


# ---------------- SHOW ATTENDANCE ----------------

@app.route("/show")
def show():

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    rows = cursor.fetchall()

    conn.close()

    return render_template("attendance.html", rows=rows)


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)