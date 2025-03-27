import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for built-in camera, 1 for external camera
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 300
counter = 0
folder = "C:/Users/Yoga/Downloads/fp/final/Data/yes"

while True:
    # Capture the frame
    success, img = cap.read()
    if not success:
        print("Error: Camera frame not captured!")
        break  # Exit if the camera is not working

    # Detect hands
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box

        h_img, w_img, _ = img.shape

        # Ensure bounding box does not go outside the image
        x1, y1 = max(x - offset, 0), max(y - offset, 0)
        x2, y2 = min(x + w + offset, w_img), min(y + h + offset, h_img)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Skipping frame: imgCrop is empty")
            continue  # Skip this frame if the cropped image is empty

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # White background

        aspectRatio = h / w

        try:
            if aspectRatio > 1:  # Height > Width
                k = imgSize / h
                wCal = math.floor(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.floor((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:  # Width >= Height
                k = imgSize / w
                hCal = math.floor(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.floor((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        except cv2.error as e:
            print("Resize error:", e)
            continue  # Skip this frame if resizing fails

        # Show Cropped and Processed Images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Show the main image with detections
    cv2.imshow("Image", img)

    # Wait for key press
    key = cv2.waitKey(10)  # Increased from 1 to 10 for stability
    if key == ord('s'):  # Save image on pressing 's'
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(f"Saved Image {counter}")

    if key == ord('q'):  # Quit on pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()
