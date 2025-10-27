import streamlit
import os
from typing import BinaryIO

def save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile, temp_dir: str) -> str:
    if uploaded_file is None:
        print("No file uploaded.")
        return ""
    
    if not uploaded_file.name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
        print("Unsupported file format. Please upload a video file.")
        return ""
    
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        print(f"Video saved to {temp_video_path}")
        return temp_video_path
    except Exception as e:
        print(f"Error saving video: {e}")
        return ""
    