import cv2

# Open a video capture object
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()

    # Show the video feed
    cv2.imshow("video_live", video_data)

    # Press 'a' key to exit the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture object and close OpenCV windows
video_cap.release()
cv2.destroyAllWindows()

# Load the face detection classifier
face_cap=cv2.CascadeClassifier("C:/Users/HP/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/cv2/data/haarcascade_frontalface_default.xml")

if face_cap.empty():
    print("Cascade Classifier not loaded")
else:
    print("Cascade Classifier loaded successfully")

# Open a new video capture object
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()

    # Convert the frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video feed with face detection
    cv2.imshow("video_live", video_data)

    # Press 'a' key to exit the loop
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture object and close OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
