import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from glob import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logreg import LogisticRegression
from mypipe import MyPipeline, FEATURE_COLUMNS

def label_from_path(image_path):
    folder = image_path.split(os.sep)[-2].lower()
    if "yawn" in folder and "no_yawn" not in folder:
        return 1
    if "no_yawn" in folder:
        return 0
    return None

def extract_landmarks_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = []
            for lm in landmarks[:468]:
                coords.extend([lm.x, lm.y, lm.z])
            return coords
    return None

def extract_dataset(image_paths, dataset_name):
    rows = []
    labels = []

    for image_path in image_paths:
        label = label_from_path(image_path)
        if label is None:
            print(f"Skipping {image_path} (unrecognized folder label)")
            continue

        coords = extract_landmarks_from_image(image_path)
        if coords is not None:
            rows.append(coords)
            labels.append(label)
        else:
            print(f"Skipping {os.path.basename(image_path)} from {dataset_name} (no face detected)")

    if len(rows) == 0:
        return None, None

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    y = np.array(labels)
    return X, y

def evaluate_on_test_set(pipe, X_test, y_test):
    predictions = np.asarray(pipe.predict(X_test)).astype(int).ravel()
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["no_yawn", "yawn"],
        zero_division=0,
    )

    print("\nISDDS test results:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"             no_yawn  yawn")
    print(f"    no_yawn     {matrix[0][0]:5d}  {matrix[0][1]:5d}")
    print(f"    yawn        {matrix[1][0]:5d}  {matrix[1][1]:5d}")
    print(report)

def train_real_model(train_images, test_images):
    print(f"YawDD train images: {len(train_images)}")
    print(f"ISDDS test images : {len(test_images)}")

    print("\nExtracting YawDD training features...")
    X_train, y_train = extract_dataset(train_images, "YawDD")
    if X_train is None:
        print("Error: Could not extract landmarks from any YawDD images.")
        return False

    print(f"YawDD feature extraction complete. Extracted {len(X_train)} samples.")

    print("\nExtracting ISDDS test features...")
    X_test, y_test = extract_dataset(test_images, "ISDDS")
    if X_test is None:
        print("Warning: No usable ISDDS test samples found. Training will continue without evaluation.")

    print("\nTraining Logistic Regression model on YawDD...")
    n_pca = min(10, X_train.shape[0], X_train.shape[1])
    model = LogisticRegression(learning_rate=0.1, max_iter=1000, C=1.0)
    pipe = MyPipeline(model, normalize=True, n_pca=n_pca)
    pipe.fit(X_train, y_train)

    if X_test is not None:
        evaluate_on_test_set(pipe, X_test, y_test)
    else:
        print("Skipped ISDDS evaluation because no test features were extracted.")

    with open("final_model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    print("\nSuccessfully trained on YawDD and saved model to final_model.pkl!")
    return True

def generate_dummy_model():
    print("YawDD dataset not found or empty. Generating a dummy model so the demo can run...")

    X_dummy = pd.DataFrame(np.random.randn(2, 1404))
    y_dummy = np.array([0, 1])

    model = LogisticRegression(learning_rate=0.1, max_iter=10)
    pipe = MyPipeline(model, normalize=False, n_pca=None)
    pipe.fit(X_dummy, y_dummy)

    with open("final_model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    print("Successfully created a dummy final_model.pkl!")

def main():
    train_images = sorted(glob(os.path.join("data", "raw", "YawDD", "*", "*.*p*")))
    test_images = sorted(glob(os.path.join("data", "raw", "ISDDS", "*", "*.*p*")))

    if len(train_images) > 0:
        success = train_real_model(train_images, test_images)
        if not success:
            generate_dummy_model()
    else:
        generate_dummy_model()

if __name__ == "__main__":
    main()