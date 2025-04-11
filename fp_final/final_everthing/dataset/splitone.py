import cv2
import os

# === CONFIG ===
video_path = r"C:\Users\Manoj\Downloads\clg material junk\Sign-Language-To-Text\fp_final\final_everthing\dataset\hello1.mp4"
output_folder = r"C:\Users\Manoj\Downloads\clg material junk\Sign-Language-To-Text\fp_final\final_everthing\dataset\hello"
chunk_duration = 17  # in seconds (this is how long each chunk will be)
frame_rate = 30  # frames per second (adjust according to your video)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === OPEN VIDEO ===
video = cv2.VideoCapture(video_path)
fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

total_duration = frame_count / fps  # Duration in seconds
print(f"Total Duration: {total_duration} seconds")

# === SPLIT INTO CHUNKS ===
frame_idx = 0
chunk_idx = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Time for the current frame
    time = frame_idx / fps

    # Split the video into chunks based on time (chunk_duration in seconds)
    if time >= chunk_idx * chunk_duration and time < (chunk_idx + 1) * chunk_duration:
        output_path = f"{output_folder}/frame_{chunk_idx:03d}_time_{int(time)}.jpg"
        cv2.imwrite(output_path, frame)  # Save the frame as an image
        print(f"Saved: {output_path}")

    # Once the time exceeds the chunk, move to the next chunk
    if time >= (chunk_idx + 1) * chunk_duration:
        chunk_idx += 1

    frame_idx += 1

# Release the video object and cleanup
video.release()
cv2.destroyAllWindows()

print("✅ Done splitting video into frames!")
