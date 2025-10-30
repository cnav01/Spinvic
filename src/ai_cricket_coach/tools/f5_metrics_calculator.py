import pandas as pd
import numpy as np
import mediapipe as mp
from typing import Dict, Any

from scripts.HELPER FUNCTIONS import calculate_angle

def extract_metrics(full_pose_csv_path: str, llm_phase_frames: Dict[str, Any], bowler_hand: str) -> Dict[str, Any]:
    """
    loads the full pose CSV file and extracts the relevant biomechanical metrics for the key frames identified by the llm.
    Args:
        full_pose_csv_path(str): path to the full pose CSV file.
        llm_phase_frames(dict): dictionary containing the key frames identified by the llm
        bowler_hand(str): 'right' or 'left' - indicates the bowling hand.
    Returns:
        Dict[str, Any]: A nested dictionary containing the extracted metrics for each key phase.
    """
    #load the full pose CSV file
    try:
        df = pd.read_csv(full_pose_csv_path)
        if df.empty:
            print(f"Error: The CSV file at {full_pose_csv_path} is empty.")
            return {}
    except FileNotFoundError:
        print(f"Error: The file at {full_pose_csv_path} was not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return {}
    
    metrics_report = {}

    is_right_arm_bowler = bowler_hand.lower() == 'right-arm'

    # Define mediapipe landmarks for the bowling arm and front leg dynamically
    b_shoulder = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    b_elbow = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.LEFT_ELBOW
    b_wrist = mp.solutions.pose.PoseLandmark.RIGHT_WRIST if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.LEFT_WRIST
    b_hip = mp.solutions.pose.PoseLandmark.RIGHT_HIP if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.LEFT_HIP

    f_hip = mp.solutions.pose.PoseLandmark.LEFT_HIP if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.RIGHT_HIP
    f_knee = mp.solutions.pose.PoseLandmark.LEFT_KNEE if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.RIGHT_KNEE
    f_ankle = mp.solutions.pose.PoseLandmark.LEFT_ANKLE if is_right_arm_bowler else mp.solutions.pose.PoseLandmark.RIGHT_ANKLE

    # other universal landmarks
    l_shoulder = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    r_shoulder = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    l_hip = mp.solutions.pose.PoseLandmark.LEFT_HIP
    r_hip = mp.solutions.pose.PoseLandmark.RIGHT_HIP
    nose = mp.solutions.pose.PoseLandmark.NOSE
    ear_left = mp.solutions.pose.PoseLandmark.LEFT_EAR
    ear_right = mp.solutions.pose.PoseLandmark.RIGHT_EAR
    
    # Helper function to get world coordinates from the existing row data
    def get_world_coords(row, landmark_enum):
        """Extracts [x, y, z] world coordinates for a given landmark from a DataFrame row."""
        base_col = f"world_lm_{landmark_enum.value}"
        if pd.isna(row[f"{base_col}_x"]) or pd.isna(row[f"{base_col}_y"]) or pd.isna(row[f"{base_col}_z"]):
            return [np.nan, np.nan, np.nan]
        return [row[f"{base_col}_x"], row[f"{base_col}_y"], row[f"{base_col}_z"]]
    
    for phase_name, frame_index in llm_phase_frames.items():
        if frame index is None or not is isinstance(frame_index, int) or frame_index < 0:
            print(f"Warning: Invalid frame index for phase '{phase_name}': {frame_index}. Skipping this phase.")
            continue

        phase_metrics = {"frame_index": frame_index}

        try:
            frame_data = df[df[frame_idx] == frame_index] #extract the row corresponding to the frame index
            if frame_data.empty:
                print(f"Warning: No data found for frame index {frame_index} in phase '{phase_name}'. Skipping this phase.")
                metrics_report[phase_name] = None
                continue
            frame_data = frame_data.iloc[0] #get the first row as a Series
        except Exception as e:
            print(f"An error occurred while extracting data for frame index {frame_index} in phase '{phase_name}': {e}")
            metrics_report[phase_name] = None
            continue

        try:
            # Extract 3D coordinates for relevant landmarks
            b_shoulder_coords = get_world_coords(frame_data, b_shoulder)
            b_elbow_coords = get_world_coords(frame_data, b_elbow)
            b_wrist_coords = get_world_coords(frame_data, b_wrist)
            b_hip_coords = get_world_coords(frame_data, b_hip)

            f_hip_coords = get_world_coords(frame_data, f_hip)
            f_knee_coords = get_world_coords(frame_data, f_knee)
            f_ankle_coords = get_world_coords(frame_data, f_ankle)

            l_shoulder_coords = get_world_coords(frame_data, l_shoulder)
            r_shoulder_coords = get_world_coords(frame_data, r_shoulder)
            l_hip_coords = get_world_coords(frame_data, l_hip)
            r_hip_coords = get_world_coords(frame_data, r_hip)
            nose_coords = get_world_coords(frame_data, nose)
            ear_left_coords = get_world_coords(frame_data, ear_left)
            ear_right_coords = get_world_coords(frame_data, ear_right)

            if np.any(np.isnan(b_shoulder_coords)) or np.any(np.isnan(b_elbow_coords)) or np.any(np.isnan(b_wrist_coords)):
                print(f"Warning: Missing bowling arm coordinates for frame index {frame_index} in phase '{phase_name}'. Skipping this phase.")
            if np.any(np.isnan(f_hip_coords)) or np.any(np.isnan(f_knee_coords)) or np.any(np.isnan(f_ankle_coords)):
                print(f"Warning: Missing front leg coordinates for frame index {frame_index} in phase '{phase_name}'. Skipping this phase.")

            # ---- Calculate Biomechanical Metrics for this specific frame ----

            
            
            
            
            # Metrics to be added!





            # ---- Calculate Biomechanical Metrics for this specific frame ----

        except Exception as e:
            print(f"Error calculating metrics for frame {frame_index} (Phase: {phase_name}): {e}")
            metrics_report[phase_name] = {"error": str(e), "frame_index": frame_index} # Record error for this phase
    print("Biomechanical metrics extraction complete.")
    return metrics_report
        