import cv2
import pickle
import numpy as np
from deepface import DeepFace
from mark_attendance import mark_attendance
import tensorflow as tf
import time

# ---------------- GPU SETUP ----------------

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ---------------- SETTINGS ----------------

EMBEDDINGS_FILE = "embeddings.pkl"
THRESHOLD = 0.5
PROCESS_EVERY = 10
REQUIRED_SECONDS = 5

marked_names = set()
frame_count = 0

last_name = "Unknown"
last_confidence = 0

confirmed_name = None
name_start_time = None

# ---------------- LOAD EMBEDDINGS ----------------

with open(EMBEDDINGS_FILE, "rb") as f:
    stored_embeddings = pickle.load(f)

# normalize stored embeddings once
for person in stored_embeddings:
    emb = np.array(stored_embeddings[person])
    stored_embeddings[person] = emb / np.linalg.norm(emb)

# ---------------- FACE DETECTOR ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit.")

# ---------------- MAIN LOOP ----------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_img = cv2.resize(face_img, (160, 160))

        if frame_count % PROCESS_EVERY == 0:

            try:

                # FIXED: removed "model=model"
                result = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet512",
                    enforce_detection=False
                )

                embedding = np.array(result[0]["embedding"])
                embedding = embedding / np.linalg.norm(embedding)

                name = "Unknown"
                min_distance = float("inf")

                for person, stored_embedding in stored_embeddings.items():

                    distance = 1 - np.dot(embedding, stored_embedding)

                    if distance < min_distance:
                        min_distance = distance
                        name = person

                # ---------------- RECOGNITION RESULT ----------------

                if min_distance > THRESHOLD:

                    last_name = "Unknown"
                    last_confidence = 0
                    confirmed_name = None
                    name_start_time = None

                else:

                    similarity = 1 - min_distance
                    last_name = name
                    last_confidence = round(similarity * 100, 2)

                    current_time = time.time()

                    if confirmed_name != name:
                        confirmed_name = name
                        name_start_time = current_time

                    else:

                        elapsed = current_time - name_start_time

                        if elapsed >= REQUIRED_SECONDS and name not in marked_names:

                            mark_attendance(name)
                            marked_names.add(name)

                            print(f"Attendance confirmed for {name}")

            except Exception as e:
                print("Recognition error:", e)

        # ---------------- DRAW FACE BOX ----------------

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        if confirmed_name == last_name and name_start_time:

            elapsed = int(time.time() - name_start_time)
            display_text = f"{last_name} ({last_confidence}%) [{elapsed}s]"

        else:

            display_text = f"{last_name} ({last_confidence}%)"

        cv2.putText(
            frame,
            display_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("DeepFace Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()