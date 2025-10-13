import cv2
import mediapipe as mp
import numpy as np
import os           
import pandas as pd
from typing import List, Dict, Any

mp_pose = mp.solutions.pose #initialize mediapipe pose components

def extract_full_pose_data(video_path: str, output_csv_path: str) -> None:
    """
    The function processes the video using mediapipe blazepose to extract all the
    landmark data and saves it to a csv file.

    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_data_rows = [] # List to hold data for each frame
    frame_count = 0

    #Defining columns for the CSV
    columns = ['frame_idx']
    for i in range(mp_pose.PoseLandmark.LEFT_FOOT_INDEX + 1):
        #normalized 2d coordinates
        columns.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z', f'lm_{i}_visibility'])
        #absolute 3d coordinates
        columns.extend([f'world_lm_{i}_x', f'world_lm_{i}_y', f'world_lm_{i}_z', f'world_lm_{i}_visibility'])

        #setting up mediapipe pose model
    with mo_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Process the image and detect the pose
            results = pose.process(image_rgb)

            # list to hold data for the current frame
            current_frame_row = [frame_count]

            if results.pose_landmarks and results.world_landmarks:
                #appending the data for each landmark
                for i in range(mp_pose.PoseLandmark.LEFT_FOOT_INDEX + 1):
                    #normalized 2d coordinates
                    lm = results.pose_landmarks.landmark[i]
                    current_frame_row.extend([lm.x, lm.y, lm.z, lm.visibility])

                    #absolute 3d coordinates
                    world_lm = results.world_landmarks.landmark[i]
                    current_frame_row.extend([world_lm.x, world_lm.y, world_lm.z, world_lm.visibility])
            else: 

