import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN

# Load the FaceNet model for face recognition
facenet = keras.models.load_model("path/to/your/facenet/model.h5")

# Load the Haar Cascade Classifiers for face, eyes, and mouth detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Load the age model (you'll need to add the path to your age estimation model)
age_model = keras.models.load_model("path/to/age_model.h5")

# Open a video capture object
video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image for FaceNet
        face_roi = cv2.resize(face_roi, (160, 160))
        face_roi = face_roi.astype("float32")
        mean, std = face_roi.mean(), face_roi.std()
        face_roi = (face_roi - mean) / std
        face_roi = np.expand_dims(face_roi, axis=0)

        # Generate the face embedding using FaceNet
        face_embedding = facenet.predict(face_roi)[0]

        # Draw a rectangle and label for the recognized face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Age estimation
        predicted_age = age_model.predict(face_embedding.reshape(1, -1))[0][0]

        # Display age on the frame
        text = f"Age: {int(predicted_age)} years"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Region of interest (ROI) for eyes and mouth within the detected face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # Detect the mouth within the face
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

    # Show the frame with face, eyes, and mouth detection
    cv2.imshow("Face, Eyes, Mouth, and Age Detection", frame)

    # Press 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
