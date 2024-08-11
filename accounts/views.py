from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import RegisterForm, LoginForm
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

CASCADE_PATH = 'C:/Users/Lenovo/Desktop/Django folder/emotion_detection/haarcascade_frontalface_default.xml'
MODEL_PATH = 'C:/Users/Lenovo/Desktop/Django folder/emotion_detection/model.h5'

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

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('emotion_detection')
            else:
                form.add_error(None, 'Invalid username or password')
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def emotion_detection_view(request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render(request, 'accounts/error.html', {'message': 'Error: Could not open video stream.'})

    while True:
        ret, frame = cap.read()
        if not ret:
            return render(request, 'accounts/error.html', {'message': 'Failed to grab frame'})

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

    return render(request, 'accounts/emotion_detection.html')
