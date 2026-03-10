import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from deepface import DeepFace

# ---------------- SETTINGS ----------------

DATASET_PATH = "dataset"
EMBEDDINGS_FILE = "embeddings.pkl"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------------- GPU SETUP ----------------

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ---------------- MAIN FUNCTION ----------------

def generate_embeddings():

    database = {}

    total_people = 0
    total_images = 0
    processed_images = 0
    skipped_images = 0

    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found.")
        return

    for person_name in os.listdir(DATASET_PATH):

        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        total_people += 1
        print(f"\nProcessing person: {person_name}")

        embeddings_list = []

        for image_name in os.listdir(person_path):

            if not image_name.lower().endswith(VALID_EXTENSIONS):
                continue

            image_path = os.path.join(person_path, image_name)
            total_images += 1

            # -------- Image Validation --------

            if os.path.getsize(image_path) == 0:
                print(f"   ✖ Skipped {image_name}: Empty file")
                skipped_images += 1
                continue

            img = cv2.imread(image_path)

            if img is None:
                print(f"   ✖ Skipped {image_name}: Corrupted image")
                skipped_images += 1
                continue

            try:

                result = DeepFace.represent(
                    img_path=image_path,
                    model_name="Facenet512",
                    detector_backend="opencv",
                    enforce_detection=False
                )

                embedding = np.array(result[0]["embedding"], dtype="float32")

                embeddings_list.append(embedding)

                processed_images += 1
                print(f"   ✔ {image_name}")

            except Exception as e:

                skipped_images += 1
                print(f"   ✖ Skipped {image_name}: {e}")

        # -------- Compute robust embedding --------

        if len(embeddings_list) > 0:

            embeddings_array = np.array(embeddings_list)

            # Use median instead of mean (robust to outliers)
            median_embedding = np.median(embeddings_array, axis=0)

            # Normalize embedding
            median_embedding = median_embedding / np.linalg.norm(median_embedding)

            database[person_name] = median_embedding

            print(f"Stored embedding for {person_name}")

        else:

            print(f"No valid images for {person_name}")

    # -------- Save embeddings --------

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(database, f)

    # -------- Statistics --------

    print("\nEmbedding generation complete")
    print("--------------------------------")
    print(f"Total persons        : {total_people}")
    print(f"Total images found   : {total_images}")
    print(f"Processed images     : {processed_images}")
    print(f"Skipped images       : {skipped_images}")
    print(f"Embeddings saved to  : {EMBEDDINGS_FILE}")


# ---------------- RUN SCRIPT ----------------

if __name__ == "__main__":
    generate_embeddings()