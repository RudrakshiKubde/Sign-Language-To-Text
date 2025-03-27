import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("sign_language_model.h5")

# Define categories
categories = ["yes", "thankyou", "hello","no","bye","perfect"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (64, 64))  # Resize to model input size
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 64, 64, 3)  # Add batch dimension

    # Predict the gesture
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_name = categories[class_index]
    confidence = np.max(predictions) * 100

    # Display result on screen
    cv2.putText(frame, f"{class_name} ({confidence:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
