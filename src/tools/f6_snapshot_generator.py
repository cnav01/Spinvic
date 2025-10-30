import cv2
import mediapipe as mp
import pandas as pd
import os
from typing import Dict, Any, List

#initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_conference=0.5)

def generate_snapshots_with_metrics(video_path: str, 
                                    full_pose_csv_path: str,
                                    llm_phase_frames: Dict[str, Any],
                                    key_Frame_metrics: Dict[str, Any],
                                    output_dir: str,
                                    bowler_hand: str

) -> List[str]:
    """
    Generates snapshots for key frames with overlaid biomechanical metrics.
    
    Args:
        video_path (str): Path to the input video file.
        full_pose_csv_path (str): Path to the full pose CSV file.
        llm_phase_frames (Dict[str, Any]): Dictionary containing key frames identified by the LLM.
        key_Frame_metrics (Dict[str, Any]): Dictionary containing metrics for each key frame.
        output_dir (str): Directory to save the output snapshots.
        bowler_hand (str): 'right' or 'left' indicating the bowling hand.
        
    Returns:
        List[str]: List of paths to the generated snapshot images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the CSV data
    try:
        df = pd.read_csv(full_pose_csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {full_pose_csv_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return []
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    generated_snapshot_paths = []

    is_right_arm_bowler = bowler_hand.lower() == 'right-arm'

    # Landmarks for metrics overlay positioning
    # Bowling arm landmarks
    b_shoulder_lm = mp_pose.PoseLandmark.RIGHT_SHOULDER if is_right_arm_bowler else mp_pose.PoseLandmark.LEFT_SHOULDER
    b_elbow_lm = mp_pose.PoseLandmark.RIGHT_ELBOW if is_right_arm_bowler else mp_pose.PoseLandmark.LEFT_ELBOW
    b_wrist_lm = mp_pose.PoseLandmark.RIGHT_WRIST if is_right_arm_bowler else mp_pose.PoseLandmark.LEFT_WRIST

    #front leg landmarks
    f_knee_lm = mp_pose.PoseLandmark.LEFT_KNEE if is_right_arm_bowler else mp_pose.PoseLandmark.RIGHT_KNEE
    f_hip_lm = mp_pose.PoseLandmark.LEFT_HIP if is_right_arm_bowler else mp_pose.PoseLandmark.RIGHT_HIP
    f_ankle_lm = mp_pose.PoseLandmark.LEFT_ANKLE if is_right_arm_bowler else mp_pose.PoseLandmark.RIGHT_ANKLE

    # Iterate through each key phase and generate snapshots
    for phase_name, frame_index in llm_phase_frames.items():
        if frame_index is None or not isinstance(frame_index, int) or frame_index < 0:
            print(f"Warning: Invalid frame index for phase '{phase_name}': {frame_index}. Skipping this phase.")
            continue

        phase_metrics = key_frame_metrics.get(phase_name)
        if not phase_metrics or "error" in phase_metrics:
            print(f"Warning: No valid metrics found for phase '{phase_name}'. Skipping this phase.")
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"Warning: Could not read frame at index {frame_index} for phase '{phase_name}'. Skipping this phase.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                if phase_metrics and "error" not in phase_metrics:

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    text_color = (255, 255, 255)
                    bg_color = (0, 0, 0)

                    def draw_text_with_background(img, text, org, font, font_scale, text_color, bg_color, thickness):
                        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        x,y = org

                        x = max(0, x)
                        y = max(text_h, y)
                        x = min(img.shape[1] - text_w, x)
                        y = min(img.shape[0], y)

                        cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), bg_color, -1)
                        cv2.putText(img, (x, y - baseline), font, font_scale, text_color, thickness, cv2.LINE_AA)

                    if b_elbow_lm.value < len(results.pose_landmarks.landmark) and \
                       b_shoulder_lm.value < len(results.pose_landmarks.landmark) and \
                       b_wrist_lm.value < len(results.pose_landmarks.landmark):
                        
                        elbow_coords = (int(results.pose_landmarks.landmark[b_elbow_lm.value].x * frame_width),
                                        int(results.pose_landmarks.landmark[b_elbow_lm.value].y * frame_height))
                        shoulder_coords = (int(results.pose_landmarks.landmark[b_shoulder_lm.value].x * frame_width),
                                           int(results.pose_landmarks.landmark[b_shoulder_lm.value].y * frame_height))
                        wrist_coords = (int(results.pose_landmarks.landmark[b_wrist_lm.value].x * frame_width),
                                        int(results.pose_landmarks.landmark[b_wrist_lm.value].y * frame_height))
                        
                        if 'bowling_elbow_angle' in phase_metrics and not pd.isna(phase_metrics['bowling_elbow_angle']):
                            draw_text_with_background(frame, f"Elbow: {phase_metrics['bowling_elbow_angle']:.1f}°",
                                                      (elbow_coords[0] + 10, elbow_coords[1]), font, font_scale, text_color, bg_color, font_thickness)
                        if 'bowling_shoulder_flexion_angle' in phase_metrics and not pd.isna(phase_metrics['bowling_shoulder_flexion_angle']):
                            draw_text_with_background(frame, f"Sh. Flex: {phase_metrics['bowling_shoulder_flexion_angle']:.1f}°",
                                                      (shoulder_coords[0] + 10, shoulder_coords[1] - 30), font, font_scale, text_color, bg_color, font_thickness)
                        # Only show wrist position at ball release, if that's the phase name
                        if phase_name == "ball_release_frame" and 'bowling_wrist_position_3d' in phase_metrics and not pd.isna(phase_metrics['bowling_wrist_position_3d'][0]):
                             wrist_pos_text = f"Wrist 3D: X:{phase_metrics['bowling_wrist_position_3d'][0]:.2f}, Y:{phase_metrics['bowling_wrist_position_3d'][1]:.2f}, Z:{phase_metrics['bowling_wrist_position_3d'][2]:.2f}"
                             draw_text_with_background(frame, wrist_pos_text, (wrist_coords[0] + 10, wrist_coords[1]), font, font_scale - 0.1, text_color, bg_color, font_thickness - 1)
                    
                    # --- Draw Front Leg Metrics ---
                    if f_knee_lm.value < len(results.pose_landmarks.landmark) and \
                       f_hip_lm.value < len(results.pose_landmarks.landmark) and \
                       f_ankle_lm.value < len(results.pose_landmarks.landmark):
                        
                        knee_coords = (int(results.pose_landmarks.landmark[f_knee_lm.value].x * frame_width),
                                       int(results.pose_landmarks.landmark[f_knee_lm.value].y * frame_height))
                        
                        if 'front_knee_angle' in phase_metrics and not pd.isna(phase_metrics['front_knee_angle']):
                            draw_text_with_background(frame, f"Knee: {phase_metrics['front_knee_angle']:.1f}°",
                                                      (knee_coords[0] + 10, knee_coords[1]), font, font_scale, text_color, bg_color, font_thickness)
                            
                    # --- Draw Trunk and Hip-Shoulder Metrics (at a fixed position or near torso) ---
                    # These are less tied to a single landmark, so a fixed corner might be better
                    if 'hip_shoulder_alignment_angle' in phase_metrics and not pd.isna(phase_metrics['hip_shoulder_alignment_angle']):
                        draw_text_with_background(frame, f"Hip-Sh. Align: {phase_metrics['hip_shoulder_alignment_angle']:.1f}°",
                                                  (50, frame_height - 100), font, font_scale, text_color, bg_color, font_thickness)
                    if 'trunk_vertical_angle' in phase_metrics and not pd.isna(phase_metrics['trunk_vertical_angle']):
                        draw_text_with_background(frame, f"Trunk Vert: {phase_metrics['trunk_vertical_angle']:.1f}°",
                                                  (50, frame_height - 60), font, font_scale, text_color, bg_color, font_thickness)
                        
            else:
                print(f"Warning: No Pose landmarks detected in frame {frame_index} for phase '{phase_name}'.")

            # Add phase label and frame number
            cv2.putText(frame, f"{phase_name.replace('_frame', '').upper()} (Frame: {frame_index})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Define output filename for the snapshot
            snapshot_filename = f"{phase_name.replace('_frame', '')}_snapshot.jpg"
            snapshot_path = os.path.join(output_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            generated_snapshot_paths.append(snapshot_path)
            print(f"SUCCESS: Snapshot for phase '{phase_name}' saved to {snapshot_path}")

        except Exception as e:
            print(f"Error generating snapshot for phase '{phase_name}' at frame {frame_index}: {e}")
            continue

    cap.release()
    pose_detector.close()
    return generated_snapshot_paths