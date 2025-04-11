from moviepy.editor import VideoFileClip

# === CONFIG ===
video_path = "hello1.mp4"
output_folder = "hello"
chunk_duration = 10  # in seconds

import os
os.makedirs(output_folder, exist_ok=True)

# === LOAD VIDEO ===
video = VideoFileClip(video_path)
total_duration = int(video.duration)

# === SPLIT INTO CHUNKS ===
for start_time in range(0, total_duration, chunk_duration):
    end_time = min(start_time + chunk_duration, total_duration)
    clip = video.subclip(start_time, end_time)
    output_path = f"{output_folder}/clip_{start_time:04d}_{end_time:04d}.mp4"
    clip.write_videofile(output_path, codec="libx264")

print("✅ Done splitting video!")
