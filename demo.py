import cv2
import mediapipe as mp
from numpy  import array
from pandas import DataFrame
from pickle import load
from time   import time
from logreg import LogisticRegression
from mypipe import MyPipeline

pipe = load(open("final_model.pkl", 'rb'))
print("Berhasil load model")
pipe.normalize = False

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No webcam found, falling back to dummy.mp4 ...")
    cap = cv2.VideoCapture("dummy.mp4")
    if not cap.isOpened():
        raise RuntimeError("No webcam and no dummy.mp4 found. Exiting.")

start_time = time()
x = 1 # displays the frame rate every 1 second
counter = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX

urutans = ["First", "Second", "Third", "Forth", "Fifth", "Sixth"]

with mp_face_mesh.FaceMesh(
    max_num_faces=6,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    image_rows, image_cols, _ = image.shape

    if results.multi_face_landmarks:
      for i, rmfl in enumerate(results.multi_face_landmarks):
        data = [(lm.x, lm.y, lm.z) for lm in rmfl.landmark]
        cord = _normalized_to_pixel_coordinates(rmfl.landmark[10].x, rmfl.landmark[10].y, image_cols, image_rows)
        cv2.putText(image, f"{urutans[i]} Face", (cord[0]-35, cord[1]-15),font, 0.5, (0, 255, 255), 1)

        for idx in rmfl.landmark:
          cord = _normalized_to_pixel_coordinates(idx.x, idx.y, image_cols, image_rows)
          cv2.putText(image, '.', cord,font, 0.3, (0, 0, 255), 2)

        data = array(data)
        data = data[:468, :]
        data = data.reshape([1, -1])
        data = DataFrame(data)
        
        prediction = pipe.predict(data)
        drowsy = int(prediction)
        text_display = f"{urutans[i].upper()} FACE TERDETEKSI MENGUAP" if drowsy else f"{urutans[i].upper()} FACE NETRAL"
        color_display = (0, 0, 255) if drowsy else (255, 0, 0)

        cv2.putText(image, 
                text_display, 
                (50, (i+1)*50), 
                font, 1, 
                color_display, 
                2, 
                cv2.LINE_4)

    # FPS
    counter+=1
    if (time() - start_time) > x :
          fps = counter / (time() - start_time)
          counter = 0
          start_time = time()
    cv2.putText(image, 
              f"FPS: {int(fps)}", 
              (50, h-50), 
              font, 1, 
              (0, 255, 0), 
              1, 
              cv2.LINE_4)
    cv2.imshow('Deteksi Kantuk', image)


    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
