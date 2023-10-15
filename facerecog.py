import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from mtcnn import MTCNN  # For face detection (you'll need to install this library)

# Load the FaceNet model (pre-trained on a large face recognition dataset)
facenet = keras.models.load_model("path/to/your/facenet/model.h5")

# Load a face detection model (MTCNN)
detector = MTCNN()

# Create a dictionary of known faces (you can replace this with your own database of faces)
known_faces = {
    "person1": np.load("path/to/person1/face_embedding.npy"),
    "person2": np.load("path/to/person2/face_embedding.npy"),
    # Add more known faces as needed
}

# Function to calculate the Euclidean distance between two face embeddings
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Open a video capture object
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()

    # Detect faces in the video frame
    faces = detector.detect_faces(video_data)

    for face in faces:
        x, y, w, h = face['box']
        x, y, w, h = abs(x), abs(y), abs(w), abs(h)
        face_roi = video_data[y:y+h, x:x+w]

        # Preprocess the face image for FaceNet
        face_roi = cv2.resize(face_roi, (160, 160))
        face_roi = face_roi.astype("float32")
        mean, std = face_roi.mean(), face_roi.std()
        face_roi = (face_roi - mean) / std
        face_roi = np.expand_dims(face_roi, axis=0)

        # Generate the face embedding using FaceNet
        face_embedding = facenet.predict(face_roi)[0]

        # Compare the face embedding to known faces
        min_dist = float('inf')
        recognized_person = None
        for name, known_face in known_faces.items():
            dist = euclidean_distance(face_embedding, known_face)
            if dist < min_dist:
                min_dist = dist
                recognized_person = name

        # Draw a rectangle and label for the recognized face
        if recognized_person:
            cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(video_data, recognized_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the video feed with face recognition
    cv2.imshow("video_live", video_data)

    # Press 'a' key to exit the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture object and close OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
