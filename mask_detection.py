import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


MODEL_PATH = "mask_detector.model"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = load_model(MODEL_PATH)

LABELS = ["No Mask", "Mask"]
COLORS = [(0, 0, 255), (0, 255, 0)]

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (128, 128))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)
        prediction = model.predict(face)[0]
        label_index = np.argmax(prediction)
        label = LABELS[label_index]
        color = COLORS[label_index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}: {prediction[label_index] * 100:.2f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
