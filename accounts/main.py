from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os
#from django.contrib import admin
#from django.urls import include, path


# Paths to files
CASCADE_PATH = '/Users/ayushibhasker/Downloads/emotion_detection/haarcascade_frontalface_default.xml'
MODEL_PATH = '/Users/ayushibhasker/Downloads/emotion_detection/model.h5'

# Load the face detection cascade and the trained model
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Cascade file not found at path: {CASCADE_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

face_classifier = cv2.CascadeClassifier(CASCADE_PATH)
classifier = load_model(MODEL_PATH)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_image(roi_gray):
    roi = roi_gray.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi

def predict_emotion(roi):
    prediction = classifier.predict(roi)[0]
    return emotion_labels[prediction.argmax()]

'''urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('Emotion_Detection.urls')),
]'''

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = preprocess_image(roi_gray)
                label = predict_emotion(roi)
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
