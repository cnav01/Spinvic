#!/usr/bin/env python3
"""
vertical_disp.py

Robust bowling analysis:
 - Hip-based vertical jump (cm) using stump calibration
 - Arm speed (rpm) computed robustly (both arms checked; right-arm preferred)
 - Front knee angle (deg)
 - Peak jump frame + airtime estimate
 - CSV export of per-frame metrics
 - Optional annotated video save (--save)

Usage:
    python vertical_disp.py --video "path/to/video.mp4" [--save]

Dependencies:
    pip install mediapipe opencv-python numpy scipy
"""

import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter

# -----------------------
# Config
# -----------------------
STUMP_HEIGHT_CM = 71.1
SMOOTH_WINDOW = 11  # must be odd for savgol
SMOOTH_POLY = 2
RPM_WINDOW_FRAMES = 20  # number of frames used to compute omega median
RPM_MIN_FRAMES = 3
CSV_OUTDIR = "output"
ANNOTATED_OUTDIR = "output_analysis"

# prefer right arm (set to 'left' to prefer left)
PREFERRED_ARM = "right"

# -----------------------
# Helpers
# -----------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def safe_compute_angle(a, b, c):
    """Compute interior angle at b formed by points a-b-c. Returns np.nan on failure."""
    try:
        a = np.array(a[:2], dtype=float)
        b = np.array(b[:2], dtype=float)
        c = np.array(c[:2], dtype=float)
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return float("nan")
        cosv = np.dot(ba, bc) / denom
        cosv = float(np.clip(cosv, -1.0, 1.0))
        return float(np.degrees(math.acos(cosv)))
    except Exception:
        return float("nan")


def compute_rpm_from_deg_series(deg_series, fps):
    """Robust RPM from degrees series using unwrap -> median(abs(diff))*fps."""
    if fps is None or fps <= 0 or len(deg_series) < RPM_MIN_FRAMES:
        return 0.0
    arr = np.array(deg_series, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.0
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr) if not np.isnan(np.nanmedian(arr)) else 0.0)
    rad = np.radians(arr)
    rad_unwrapped = np.unwrap(rad)
    omega = np.diff(rad_unwrapped) * fps  # rad/s per frame
    if len(omega) == 0:
        return 0.0
    seg = omega[-min(len(omega), RPM_WINDOW_FRAMES):]
    mean_omega = float(np.nanmedian(np.abs(seg)))
    rpm = (mean_omega * 60.0) / (2 * math.pi)
    return rpm


def draw_panel(frame, text, value_str, top_left, size=(300, 70), bar_frac=0.0):
    """Draw a small panel. bar_frac may be NaN; it's clamped safely to [0,1]."""
    x, y = top_left
    w, h = size
    # background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)
    cv2.putText(frame, text, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, value_str, (x + 10, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)

    # safe bar_frac
    try:
        if bar_frac is None or np.isnan(bar_frac):
            bar_frac = 0.0
    except Exception:
        bar_frac = 0.0
    bar_frac = float(min(max(bar_frac, 0.0), 1.0))
    # draw vertical bar on right
    bar_w = int((w - 12) * bar_frac)
    cv2.rectangle(frame, (x + w - 12, y + 8), (x + w - 4, y + h - 8), (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (x + w - 12, y + 8), (x + w - 12 + max(1, bar_w), y + h - 8), (120, 220, 120), -1)


# -----------------------
# Main processing
# -----------------------
def main(args):
    os.makedirs(CSV_OUTDIR, exist_ok=True)
    os.makedirs(ANNOTATED_OUTDIR, exist_ok=True)

    video_path = args.video
    save_video = args.save

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: cannot open", video_path)
        return

    # read first frame for stump clicks
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: cannot read first frame")
        return
    fh, fw = first_frame.shape[:2]

    stump_pts = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            stump_pts.append((x, y))
            print("Selected point", len(stump_pts), ":", (x, y))

    cv2.namedWindow("Select stump top and bottom - click 2 points")
    cv2.setMouseCallback("Select stump top and bottom - click 2 points", on_click)

    print("Click on the TOP of a stump, then the BOTTOM of the same stump (2 clicks).")
    while True:
        vis = first_frame.copy()
        for i, pt in enumerate(stump_pts):
            cv2.circle(vis, pt, 6, (0, 255, 0), -1)
            cv2.putText(vis, f"{i+1}", (pt[0] + 6, pt[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Select stump top and bottom - click 2 points", vis)
        k = cv2.waitKey(10) & 0xFF
        if len(stump_pts) >= 2:
            break
        if k == 27:
            print("Calibration aborted.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Select stump top and bottom - click 2 points")
    (sx1, sy1), (sx2, sy2) = stump_pts[:2]
    stump_px = float(np.hypot(sx2 - sx1, sy2 - sy1))
    cm_per_px = STUMP_HEIGHT_CM / stump_px if stump_px > 0 else 0.0
    print(f"Stump pixel distance: {stump_px:.2f}px -> scale: {cm_per_px:.4f} cm/px")

    # OPTIONAL: prepare VideoWriter
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(ANNOTATED_OUTDIR, base_name + "_annotated.mp4")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (fw, fh))
        print("Annotated video will be saved to:", out_path)

    # Pose
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True, enable_segmentation=False)

    # Storage for per-frame metrics
    frame_idx = 0
    timestamps = []
    hip_y_px_series = []
    hip_x_series = []
    arm_left_angles = []
    arm_right_angles = []
    knee_left_angles = []
    knee_right_angles = []
    frame_indices = []
    chosen_rpm_series = []

    fps_est = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # if cap fps is zero, we'll estimate using realtime.
    use_time_fps = (fps_est <= 0.1)
    t0 = time.time()
    prev_time = t0

    print("Processing... press ESC to stop early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        now = time.time()
        dt = now - prev_time if (now - prev_time) > 1e-6 else 1.0 / 30.0
        prev_time = now
        if use_time_fps:
            # estimate fps as median over last few frames
            if frame_idx == 1:
                fps = 30.0
            else:
                fps = 1.0 / dt
        else:
            fps = fps_est

        timestamps.append((frame_idx - 1) / fps if fps > 0 else 0.0)
        frame_indices.append(frame_idx - 1)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        # default values
        hip_y_px = float("nan")
        hip_x_px = float("nan")
        left_elbow_deg = float("nan")
        right_elbow_deg = float("nan")
        left_knee_deg = float("nan")
        right_knee_deg = float("nan")

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            # get pixel coords helper
            def to_px(lm_item):
                return (lm_item.x * fw, lm_item.y * fh, lm_item.z if hasattr(lm_item, "z") else 0.0)

            # hips midpoint
            lhip = to_px(lm[mp_pose.PoseLandmark.LEFT_HIP.value])
            rhip = to_px(lm[mp_pose.PoseLandmark.RIGHT_HIP.value])
            hip_mid = ((lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0)
            hip_x_px, hip_y_px = hip_mid

            # left arm
            l_sh = to_px(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            l_el = to_px(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value])
            l_wr = to_px(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])
            left_elbow_deg = safe_compute_angle(l_sh, l_el, l_wr)

            # right arm
            r_sh = to_px(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            r_el = to_px(lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
            r_wr = to_px(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
            right_elbow_deg = safe_compute_angle(r_sh, r_el, r_wr)

            # knees (hip-knee-ankle)
            l_knee = to_px(lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
            l_ank = to_px(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
            r_knee = to_px(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
            r_ank = to_px(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            left_knee_deg = safe_compute_angle(lhip, l_knee, l_ank)
            right_knee_deg = safe_compute_angle(rhip, r_knee, r_ank)

            # draw landmarks
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1, circle_radius=1))

        # append series (use nan if missing)
        hip_y_px_series.append(float(hip_y_px))
        hip_x_series.append(float(hip_x_px))
        arm_left_angles.append(float(left_elbow_deg))
        arm_right_angles.append(float(right_elbow_deg))
        knee_left_angles.append(float(left_knee_deg))
        knee_right_angles.append(float(right_knee_deg))

        # compute RPMs
        rpm_left = compute_rpm_from_deg_series(arm_left_angles, fps)
        rpm_right = compute_rpm_from_deg_series(arm_right_angles, fps)

        # choose arm: prefer PREFERRED_ARM if it has reasonable rpm, otherwise pick higher rpm
        if PREFERRED_ARM.lower().startswith("r"):
            preferred_rpm = rpm_right
        else:
            preferred_rpm = rpm_left
        # choose the larger unless preferred has tiny signal
        if preferred_rpm < 1.0 and max(rpm_left, rpm_right) > preferred_rpm:
            chosen_rpm = max(rpm_left, rpm_right)
        else:
            chosen_rpm = preferred_rpm

        chosen_rpm_series.append(chosen_rpm)

        # Compute smoothed hip series for jump calc (only if enough frames)
        hip_arr = np.array([v for v in hip_y_px_series if not np.isnan(v)])
        if len(hip_arr) >= 5:
            # optionally smooth hip positions to reduce jitter
            try:
                if len(hip_arr) >= SMOOTH_WINDOW and SMOOTH_WINDOW % 2 == 1:
                    hip_smooth = savgol_filter(hip_arr, SMOOTH_WINDOW, SMOOTH_POLY)
                else:
                    hip_smooth = hip_arr
            except Exception:
                hip_smooth = hip_arr
            # compute baseline ground (max) and peak (min)
            # we use the extremes across the sequence observed so far
            y_ground_px = float(np.max(hip_smooth))
            y_min_px = float(np.min(hip_smooth))
            jump_px = max(0.0, y_ground_px - y_min_px)
            jump_cm = jump_px * cm_per_px
        else:
            y_ground_px = float("nan")
            y_min_px = float("nan")
            jump_px = 0.0
            jump_cm = 0.0

        # Estimate airtime: find frames around peak where hip is elevated above a threshold
        airtime_s = 0.0
        peak_frame = None
        ground_frame = None
        try:
            if len(hip_arr) >= 5 and jump_px > 1.0:
                # find indices in hip_y_px_series corresponding to hip_smooth extremes
                full = np.array(hip_y_px_series, dtype=float)
                valid_idx = np.where(~np.isnan(full))[0]
                if valid_idx.size > 0:
                    full_valid = full[valid_idx]
                    # peak index relative to valid_idx
                    rel_peak = int(np.argmin(full_valid))
                    rel_ground = int(np.argmax(full_valid))
                    peak_frame = int(valid_idx[rel_peak])
                    ground_frame = int(valid_idx[rel_ground])
                    # threshold low enough to say "in-air" (10% of jump)
                    thr = y_min_px + 0.10 * jump_px
                    # find first frame before peak where hip crosses above thr (takeoff)
                    # and first frame after peak where hip goes back below thr (landing)
                    # work on valid_idx/frame sequence
                    vals = full_valid
                    # indices before/after
                    before_peak = valid_idx[:rel_peak + 1]
                    after_peak = valid_idx[rel_peak:]
                    takeoff_idx = None
                    landing_idx = None
                    # takeoff: last index before peak where val >= thr (i.e., lower or equal)
                    for i_rel in range(len(before_peak) - 1, -1, -1):
                        if full_valid[i_rel] >= thr:
                            takeoff_idx = int(before_peak[i_rel])
                            break
                    # landing: first index after peak where val >= thr
                    for i_rel in range(0, len(after_peak)):
                        if full_valid[rel_peak + i_rel] >= thr:
                            landing_idx = int(after_peak[i_rel])
                            break
                    if takeoff_idx is not None and landing_idx is not None and landing_idx > takeoff_idx:
                        airtime_s = (landing_idx - takeoff_idx) / fps if fps > 0 else 0.0
        except Exception:
            airtime_s = 0.0

        # prepare overlays
        arm_str = f"{chosen_rpm:5.1f} rpm" if chosen_rpm > 0 else "— rpm"
        # pick front knee as the one with the greater forward x (heuristic) using hip_x
        front_knee_deg = float("nan")
        if not np.isnan(hip_x_px) and len(hip_x_series) >= 5:
            # heuristics: compare mean ankle x over recent frames to decide front leg; fallback choose right
            try:
                # use last known knee values
                # we will assume front leg is the one with smaller x if bowler faces left, or larger if faces right
                # to keep simple: choose the knee with smaller angle variance? fallback to right
                # For now, choose right knee as front (user said right-arm)
                if PREFERRED_ARM.lower().startswith("r"):
                    front_knee_deg = knee_right_angles[-1]
                else:
                    front_knee_deg = knee_left_angles[-1]
            except Exception:
                front_knee_deg = knee_right_angles[-1] if len(knee_right_angles) > 0 else float("nan")
        knee_str = f"{(front_knee_deg if not np.isnan(front_knee_deg) else 0.0):5.1f}°" if not np.isnan(front_knee_deg) else "— °"
        jump_str = f"{jump_cm:5.2f} cm" if jump_cm > 0 else "— cm"

        # sample fractions for bars
        arm_frac = min(chosen_rpm / 400.0, 1.0) if not np.isnan(chosen_rpm) else 0.0
        knee_frac = min((front_knee_deg or 0.0) / 140.0, 1.0) if not np.isnan(front_knee_deg) else 0.0
        jump_frac = min(jump_cm / 60.0, 1.0) if jump_cm is not None else 0.0

        draw_panel(frame, "Arm Speed", arm_str, (20, 20), size=(320, 70), bar_frac=arm_frac)
        draw_panel(frame, "Front Knee Angle", knee_str, (20, 110), size=(320, 70), bar_frac=knee_frac)
        draw_panel(frame, "Vertical Jump", jump_str, (20, 200), size=(320, 70), bar_frac=jump_frac)

        # annotate scale and frame/time
        cv2.line(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (0, 200, 255), 2)
        cv2.putText(frame, f"Scale: {cm_per_px:.4f} cm/px", (20, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(frame, f"Frame: {frame_idx}  Time: {timestamps[-1]:.2f}s", (fw - 380, fh - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        # if peak detected, annotate
        if peak_frame is not None and frame_idx - 1 == peak_frame:
            cv2.putText(frame, f"PEAK JUMP ({jump_cm:.2f} cm)", (fw//2 - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 3)

        cv2.imshow("Bowling Insights (press ESC to stop)", frame)
        if save_video and writer is not None:
            writer.write(frame)

        # escape
        if cv2.waitKey(1) & 0xFF == 27:
            print("Interrupted by user.")
            break

    # close
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    pose.close()

    # -----------------------
    # Post-process & export CSV + final summary
    # -----------------------
    # compute final jump stats using all (smoothed) hip values
    full_hip = np.array([v for v in hip_y_px_series if not np.isnan(v)], dtype=float)
    if full_hip.size >= 3:
        try:
            if full_hip.size >= SMOOTH_WINDOW and SMOOTH_WINDOW % 2 == 1:
                full_hip_s = savgol_filter(full_hip, SMOOTH_WINDOW, SMOOTH_POLY)
            else:
                full_hip_s = full_hip
        except Exception:
            full_hip_s = full_hip
        y_ground_px_final = float(np.max(full_hip_s))
        y_min_px_final = float(np.min(full_hip_s))
        jump_px_final = max(0.0, y_ground_px_final - y_min_px_final)
        jump_cm_final = jump_px_final * cm_per_px
        peak_idx_final = int(np.argmin(full_hip_s))
        ground_idx_final = int(np.argmax(full_hip_s))
    else:
        jump_cm_final = 0.0
        peak_idx_final = None
        ground_idx_final = None

    # compute median chosen_rpm
    if len(chosen_rpm_series) > 0:
        median_rpm = float(np.nanmedian(chosen_rpm_series))
    else:
        median_rpm = 0.0

    # CSV export: per-frame values
    csv_file = os.path.join(CSV_OUTDIR, "metrics.csv")
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "hip_y_px", "arm_left_deg", "arm_right_deg", "chosen_rpm", "knee_left_deg", "knee_right_deg"])
        for i in range(len(frame_indices)):
            hipv = hip_y_px_series[i] if i < len(hip_y_px_series) else ""
            al = arm_left_angles[i] if i < len(arm_left_angles) else ""
            ar = arm_right_angles[i] if i < len(arm_right_angles) else ""
            cr = chosen_rpm_series[i] if i < len(chosen_rpm_series) else ""
            kl = knee_left_angles[i] if i < len(knee_left_angles) else ""
            kr = knee_right_angles[i] if i < len(knee_right_angles) else ""
            w.writerow([frame_indices[i], timestamps[i], hipv, al, ar, cr, kl, kr])

    print("\n--- Summary ---")
    print(f"Video: {video_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Estimated vertical jump (hip-based): {jump_cm_final:.2f} cm")
    if peak_idx_final is not None:
        print(f"Peak hip index (relative cleaned series): {peak_idx_final} (see metrics.csv for frame mapping)")
    print(f"Estimated arm speed (median over video): {median_rpm:.1f} rpm")
    # show a last front knee value
    last_knee = None
    if PREFERRED_ARM.lower().startswith("r"):
        last_knee = knee_right_angles[-1] if len(knee_right_angles) > 0 else float("nan")
    else:
        last_knee = knee_left_angles[-1] if len(knee_left_angles) > 0 else float("nan")
    if not np.isnan(last_knee):
        print(f"Front knee angle at last frame: {last_knee:.1f} °")
    print(f"Airtime estimate (simple heuristic): {airtime_s:.3f} s")
    print(f"CSV metrics saved to: {csv_file}")
    if save_video:
        print(f"Annotated video saved to: {out_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bowling vertical disp + arm rpm + knee angle analysis")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--save", action="store_true", help="Save annotated video to output_analysis/")
    args = parser.parse_args()
    main(args)
