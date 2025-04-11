import cv2
import os

# === CONFIG ===
video_path = r"C:/Users/Yoga/Downloads/fp/final/fp_final/final_everthing/dataset/iloveyou1.mp4"
output_folder = r"C:/Users/Yoga/Downloads/fp/final/fp_final/final_everthing/dataset/iloveyou"
desired_frame_count = 900

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("❌ Failed to open video. Check the path.")
    exit()

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, total_frames // desired_frame_count)

print(f"Total frames in video: {total_frames}")
print(f"Extracting 1 frame every {step} frames.")

frame_idx = 0
saved_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    if frame_idx % step == 0 and saved_count < desired_frame_count:
        output_path = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
        saved_count += 1

    frame_idx += 1

video.release()
cv2.destroyAllWindows()

print("✅ Done! Extracted", saved_count, "frames.")
