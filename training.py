import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define dataset path
data_dir = "Data"
categories = ["yes", "thankyou", "hello","no","bye","perfect"]

# Image parameters
img_size = 64  # Resize images to 64x64 pixels
X, y = [], []  # Data and labels

# Load images and assign labels
for label, category in enumerate(categories):
    path = os.path.join(data_dir, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))  # Resize image
        X.append(img)
        y.append(label)  # Assign label

# Convert lists to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixel values
y = to_categorical(y, num_classes=len(categories))  # One-hot encoding

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test), batch_size=32)

# Save the trained model
model.save("sign_language_model.h5")

print("Model training complete. Saved as 'sign_language_model.h5'")
