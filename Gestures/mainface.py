import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
from deepface import DeepFace

face_classifier = cv2.CascadeClassifier()
face_classifier.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))

cap = cv2.VideoCapture(0)  # Adjust the index if your camera is not on index 0

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray)
    
    for (x, y, w, h) in faces:
        face_img = frame_gray[y:y+h, x:x+w]
        response = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
        dominant_emotion = response["dominant_emotion"]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
