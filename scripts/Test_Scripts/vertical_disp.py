
"""
bowling_insights.py

Usage:
    python bowling_insights.py --video path/to/video.mp4

Functionality:
 - Side-view video expected.
 - On program start: the first frame is shown and you must click two points:
   1) top of a visible stump
   2) bottom of that stump
 - The script computes a pixel->cm scale using stump height = 71.1 cm.
 - Then it processes the video with MediaPipe Pose and overlays:
    * Arm Speed (rpm)
    * Front Knee Angle (deg)
    * Vertical Jump (cm)
 - It displays an OpenCV window with the skeleton + metric panels.

Notes:
 - If landmarks fail for a frame, that frame is skipped for metric computations.
 - Uses hip midpoint for jump height estimation (smoothed).
"""

import argparse
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

# ---------- Config ----------
STUMP_HEIGHT_CM = 71.1
SMOOTH_WINDOW = 11   # for Savitzky-Golay smoothing (must be odd)
SMOOTH_POLY = 2
FPS_SMOOTH_LEN = 5   # number of frames to average for fps estimate
ANGLE_HISTORY = 15   # used to compute recent angular velocity
ARM_RPM_SMOOTH = 5   # smoothing for rpm display
JUMP_BASELINE_FRAMES = 30  # first N frames to compute baseline ground hip y
# ----------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# global to store two clicks for stump points
stump_pts = []


def click_event(event, x, y, flags, param):
    global stump_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        stump_pts.append((x, y))
        print(f"Selected point {len(stump_pts)}: {(x,y)}")


def compute_angle(a, b, c):
    """
    Angle at point b formed by a-b-c in degrees.
    a,b,c are (x,y,z) or (x,y) numpy arrays.
    """
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle


def get_landmark_xy(landmark, frame_w, frame_h):
    """Return (x_px, y_px) from normalized landmark."""
    return int(landmark.x * frame_w), int(landmark.y * frame_h)


def vector_2d(a, b):
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    return b - a


def smooth_series(arr, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    if len(arr) < window:
        return np.array(arr)
    try:
        return savgol_filter(np.array(arr), window, poly)
    except Exception:
        return np.array(arr)


def draw_panel(frame, text, value_str, top_left, size=(300, 70), bar_frac=0.5, color=(50, 50, 50)):
    x, y = top_left
    w, h = size
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)  # panel bg
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(frame, text, (x + 8, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    # value big
    cv2.putText(frame, value_str, (x + 10, y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
    # simple progress bar indicating magnitude
    if bar_frac is None or np.isnan(bar_frac):
        bar_frac = 0.0
    bar_w = int(w * min(max(bar_frac, 0.0), 1.0))

    cv2.rectangle(frame, (x + w - 12, y + 8), (x + w - 4, y + h - 8), (60, 60, 60), -1)
    cv2.rectangle(frame, (x + w - 12, y + 8), (x + w - 12 + bar_w // 10, y + h - 8), (120, 220, 120), -1)


def main(args):
    global stump_pts

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    # read first frame to pick stump points and to get frame dims
    ret, first_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame.")
        return

    frame_h, frame_w = first_frame.shape[:2]
    clone = first_frame.copy()
    cv2.namedWindow("Select stump top and bottom - click 2 points")
    cv2.setMouseCallback("Select stump top and bottom - click 2 points", click_event)

    print("Click on the TOP of a stump, then the BOTTOM of the same stump (2 clicks).")
    while True:
        display = clone.copy()
        # draw existing points
        for i, pt in enumerate(stump_pts):
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.putText(display, f"{i+1}", (pt[0]+6, pt[1]+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Select stump top and bottom - click 2 points", display)
        key = cv2.waitKey(1) & 0xFF
        if len(stump_pts) >= 2:
            break
        if key == 27:  # ESC to quit
            print("Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select stump top and bottom - click 2 points")
    (sx1, sy1), (sx2, sy2) = stump_pts[:2]
    stump_px = float(np.hypot(sx2 - sx1, sy2 - sy1))
    if stump_px <= 0.0:
        print("Invalid stump selection.")
        return
    cm_per_px = STUMP_HEIGHT_CM / stump_px
    print(f"Stump pixel distance: {stump_px:.2f}px -> scale: {cm_per_px:.4f} cm/px")

    # Reset capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # storage for time-series
    hip_y_px_series = []
    ankle_left_y = []
    ankle_right_y = []
    hip_x_series = []
    ankle_left_x = []
    ankle_right_x = []
    arm_angles = []  # shoulder-elbow-wrist angle (deg)
    knee_left_angles = []
    knee_right_angles = []
    fps_estimates = deque(maxlen=FPS_SMOOTH_LEN)
    prev_time = None

    # small deque to compute RPM and smooth it
    recent_arm_angles = deque(maxlen=ANGLE_HISTORY)
    recent_rpm = deque(maxlen=ARM_RPM_SMOOTH)

    # we'll detect which leg is front by analyzing ankle x positions over first few frames
    ankle_x_initial = []

    pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)

    frame_count = 0
    start_time = time.time()
    last_display_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_frame = time.time()
        frame_count += 1

        if prev_time is None:
            prev_time = t_frame
        dt = t_frame - prev_time if (t_frame - prev_time) > 0 else 1.0 / 30.0
        prev_time = t_frame
        fps_estimates.append(1.0 / dt)
        fps = np.mean(fps_estimates) if len(fps_estimates) > 0 else cap.get(cv2.CAP_PROP_FPS) or 30.0

        # process pose
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # convert to px coords
            # indices: left hip=23, right hip=24, left ankle=27, right ankle=28
            # left shoulder=11, left elbow=13, left wrist=15
            # right shoulder=12, right elbow=14, right wrist=16
            try:
                left_hip = (lm[23].x * frame_w, lm[23].y * frame_h)
                right_hip = (lm[24].x * frame_w, lm[24].y * frame_h)
                hip_mid = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)

                left_ankle = (lm[27].x * frame_w, lm[27].y * frame_h)
                right_ankle = (lm[28].x * frame_w, lm[28].y * frame_h)

                # store series
                hip_y_px_series.append(hip_mid[1])
                hip_x_series.append(hip_mid[0])
                ankle_left_y.append(left_ankle[1])
                ankle_right_y.append(right_ankle[1])
                ankle_left_x.append(left_ankle[0])
                ankle_right_x.append(right_ankle[0])

                # For initial frames, store ankle x to guess front leg
                if len(ankle_x_initial) < 15:
                    ankle_x_initial.append((left_ankle[0], right_ankle[0]))

                # Arm angle: compute angle at elbow for the arm that is more active / in front.
                # We'll compute both left and right elbow angles and pick the larger angular speed arm.
                left_shoulder = (lm[11].x * frame_w, lm[11].y * frame_h)
                left_elbow = (lm[13].x * frame_w, lm[13].y * frame_h)
                left_wrist = (lm[15].x * frame_w, lm[15].y * frame_h)

                right_shoulder = (lm[12].x * frame_w, lm[12].y * frame_h)
                right_elbow = (lm[14].x * frame_w, lm[14].y * frame_h)
                right_wrist = (lm[16].x * frame_w, lm[16].y * frame_h)

                left_elbow_angle = compute_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = compute_angle(right_shoulder, right_elbow, right_wrist)
                # choose the arm with larger instantaneous motion (we'll choose whichever has greater change)
                arm_angles.append((left_elbow_angle, right_elbow_angle))

                # knee angles
                # left knee: hip(23)-knee(25)-ankle(27)
                left_knee = (lm[25].x * frame_w, lm[25].y * frame_h)
                right_knee = (lm[26].x * frame_w, lm[26].y * frame_h)
                left_knee_angle = compute_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = compute_angle(right_hip, right_knee, right_ankle)
                knee_left_angles.append(left_knee_angle)
                knee_right_angles.append(right_knee_angle)

                # draw landmarks and skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1, circle_radius=1))

            except Exception as e:
                # landmark indexing errors
                # skip frame
                print("Landmark error:", e)
                continue

        # After collecting some frames, determine which leg is front (x smaller or larger depends on direction)
        front_leg = None  # 'left' or 'right'
        if len(ankle_x_initial) >= 10 and front_leg is None:
            # compute average x for left and right ankles
            arr = np.array(ankle_x_initial)
            left_mean = np.mean(arr[:, 0])
            right_mean = np.mean(arr[:, 1])
            # In side view, front leg is the one that is further forward in x.
            # The direction (left->right) depends on video; we'll assume forward = larger x (rightwards).
            # To be safer, check which ankle is more forward relative to hips average direction:
            # We'll choose the ankle with greater mean x as front if x variance shows forward motion.
            if abs(left_mean - right_mean) > 5.0:  # pixels
                front_leg = 'left' if left_mean > right_mean else 'right'
            else:
                front_leg = 'left'  # fallback

        # Now compute metrics if we have enough series
        display_arm_rpm = 0.0
        display_knee_angle = float('nan')
        display_jump_cm = 0.0

        # ARM SPEED: take elbow angle time series for selected arm
        if len(arm_angles) >= 4:
            # choose left/right per-frame based on which shows larger delta on average recently
            # convert arm_angles list of tuples -> two arrays
            arr = np.array(arm_angles)
            left_series = arr[:, 0].astype(np.float64)
            right_series = arr[:, 1].astype(np.float64)

            # smooth a little
            left_smooth = smooth_series(left_series, window=min(SMOOTH_WINDOW, max(3, len(left_series) if len(left_series)%2==1 else len(left_series)-1)))
            right_smooth = smooth_series(right_series, window=min(SMOOTH_WINDOW, max(3, len(right_series) if len(right_series)%2==1 else len(right_series)-1)))

            # compute recent mean absolute angular velocity (deg/s)
            if len(left_smooth) >= 3:
                left_d = np.abs(np.diff(left_smooth)) * fps  # deg/s
                right_d = np.abs(np.diff(right_smooth)) * fps
                # average over last few diffs
                left_deg_s = np.mean(left_d[-min(len(left_d), 6):]) if len(left_d) > 0 else 0.0
                right_deg_s = np.mean(right_d[-min(len(right_d), 6):]) if len(right_d) > 0 else 0.0

                # pick arm with larger angular speed (likely bowling arm)
                chosen_deg_s = left_deg_s if left_deg_s > right_deg_s else right_deg_s
                # deg/s -> rpm: rpm = (deg/s) * (1 rotation / 360 deg) * 60 s/min
                rpm = chosen_deg_s * (60.0 / 360.0)
                # smoothing
                recent_rpm.append(rpm)
                display_arm_rpm = np.mean(recent_rpm)

        # KNEE ANGLE: front knee angle at the current/latest frame
        if front_leg is not None and len(knee_left_angles) > 0 and len(knee_right_angles) > 0:
            if front_leg == 'left':
                display_knee_angle = knee_left_angles[-1]
            else:
                display_knee_angle = knee_right_angles[-1]

        # VERTICAL JUMP:
        # We'll compute baseline ground hip y as mean of first N frames' hip y (or initial frames where hip exists).
        if len(hip_y_px_series) >= 3:
            hip_arr = np.array(hip_y_px_series)
            # smooth hip series
            if len(hip_arr) >= SMOOTH_WINDOW:
                hip_smooth = smooth_series(hip_arr, window=SMOOTH_WINDOW)
            else:
                hip_smooth = hip_arr

            # baseline: mean of first JUMP_BASELINE_FRAMES frames (if available)
            baseline_frames = min(JUMP_BASELINE_FRAMES, len(hip_smooth))
            y_ground_px = np.mean(hip_smooth[:baseline_frames])
            y_min_px = np.min(hip_smooth)  # highest body point (smallest y px)
            jump_px = max(0.0, y_ground_px - y_min_px)
            display_jump_cm = jump_px * cm_per_px

        # Draw metric panels on frame (top-left panels)
        # Format strings
        arm_str = f"{display_arm_rpm:5.0f} rpm" if display_arm_rpm > 0 else "— rpm"
        knee_str = f"{display_knee_angle:5.1f}°" if not math.isnan(display_knee_angle) else "— °"
        jump_str = f"{display_jump_cm:5.1f} cm" if display_jump_cm > 0 else "— cm"

        # sample bar fraction for visualization (clamped)
        # arm: assume 0-400 rpm mapping
        arm_frac = min(display_arm_rpm / 400.0, 1.0)
        # knee: 0-140 degrees (0 fully straight)
        knee_frac = min((display_knee_angle or 0.0) / 140.0, 1.0)
        # jump: 0-60 cm mapping
        jump_frac = min(display_jump_cm / 60.0, 1.0)

        # panels
        draw_panel(frame, "Arm Speed", arm_str, (20, 20), size=(300, 70), bar_frac=arm_frac)
        if not np.isnan(knee_frac):
            draw_panel(frame, "Front Knee Angle", knee_str, (20, 110), size=(300, 70), bar_frac=knee_frac)
        else:
            draw_panel(frame, "Front Knee Angle", "N/A", (20, 110), size=(300, 70), bar_frac=0.0)

        draw_panel(frame, "Vertical Jump", jump_str, (20, 200), size=(300, 70), bar_frac=jump_frac)

        # annotate stump selection on frame
        cv2.line(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 200, 255), 2)
        cv2.putText(frame, f"Scale: {cm_per_px:.4f} cm/px", (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        # show frame
        cv2.imshow("Bowling Insights", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    # end while
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # final summary printout
    print("\n--- Summary ---")
    if len(hip_y_px_series) > 0:
        print(f"Estimated vertical jump (max): {display_jump_cm:.2f} cm")
    if display_arm_rpm > 0:
        print(f"Estimated arm speed (recent average): {display_arm_rpm:.1f} rpm")
    if not math.isnan(display_knee_angle):
        print(f"Front knee angle at last frame: {display_knee_angle:.1f} °")
    print("Processing complete.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract arm speed, knee angle, vertical jump from bowling video using MediaPipe Pose.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    args = parser.parse_args()
    main(args)
