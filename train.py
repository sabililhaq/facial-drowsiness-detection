import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from glob import glob

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logreg import LogisticRegression
from mypipe import MyPipeline

def extract_landmarks_from_image(image_path):
    """
    Reads an image, uses MediaPipe FaceMesh to extract 468 landmark coordinates (x, y, z).
    Returns a list of 1404 floats (468 * 3), or None if no face is detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            # Extract first 468 landmarks (exclude iris coordinates)
            coords = []
            for lm in landmarks[:468]:
                coords.extend([lm.x, lm.y, lm.z])
            return coords
    return None

def train_real_model(train_images, test_images):
    print(f"Dataset found! Extracting features from {len(train_images)} training and {len(test_images)} test images...")
    
    train_data = []
    train_labels = []
    
    # Process training images
    for im_path in train_images:
        # Determine label from folder name (yawn -> 1, no_yawn/neutral -> 0)
        folder = im_path.split(os.sep)[-2].lower()
        label = 1 if 'yawn' in folder and 'no_yawn' not in folder else 0
        
        coords = extract_landmarks_from_image(im_path)
        if coords is not None:
            train_data.append(coords)
            train_labels.append(label)
        else:
            print(f"Skipping {os.path.basename(im_path)} (no face detected)")
            
    if len(train_data) == 0:
        print("Error: Could not extract landmarks from any images.")
        return False
        
    print(f"Feature extraction complete. Extracted {len(train_data)} samples.")
    
    # Create DataFrame
    columns = []
    for i in range(468):
        columns.extend([f"x{i+1}", f"y{i+1}", f"z{i+1}"])
        
    X_train = pd.DataFrame(train_data, columns=columns)
    y_train = np.array(train_labels)
    
    print("Training Logistic Regression Model with Pipeline...")
    # Best default parameters (e.g. max_iter=1000, learning_rate=0.1, C=1.0)
    model = LogisticRegression(learning_rate=0.1, max_iter=1000, C=1.0)
    # We set normalize=True and PCA with 10 components for training
    # Since demo.py sets pipe.normalize = False at inference, we'll follow that structure.
    pipe = MyPipeline(model, normalize=True, n_pca=10)
    
    pipe.fit(X_train, y_train)
    
    # Save the trained model
    with open("final_model.pkl", "wb") as f:
        pickle.dump(pipe, f)
        
    print("Successfully trained and saved model to final_model.pkl!")
    return True

def generate_dummy_model():
    print("Dataset not found or empty. Generating a dummy model to allow the demo to run...")
    
    # Create 2 dummy samples (one yawn, one neutral) with 1404 features
    X_dummy = pd.DataFrame(np.random.randn(2, 1404))
    y_dummy = np.array([0, 1])
    
    model = LogisticRegression(learning_rate=0.1, max_iter=10)
    pipe = MyPipeline(model, normalize=False, n_pca=None)
    pipe.fit(X_dummy, y_dummy)
    
    with open("final_model.pkl", "wb") as f:
        pickle.dump(pipe, f)
        
    print("Successfully created a dummy final_model.pkl!")

def main():
    # Look for raw datasets
    train_images = sorted(glob(os.path.join("data", "raw", "YawDD", "*", "*.*p*")))
    test_images  = sorted(glob(os.path.join("data", "raw", "ISDDS", "*", "*.*p*")))
    
    if len(train_images) > 0:
        success = train_real_model(train_images, test_images)
        if not success:
            generate_dummy_model()
    else:
        generate_dummy_model()

if __name__ == "__main__":
    main()
