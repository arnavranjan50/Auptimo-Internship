import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from scipy.signal import butter, filtfilt
from rtmlib.tools.solution.pose_tracker import PoseTracker
from rtmlib.tools.solution.body import Body
from rtmlib.visualization.draw import draw_skeleton

CONFIG = {
    "video_path": "WhatsApp Video 2026-06-22 at 18.24.47.mp4",
    "data_save_path": "gait_data1.csv",
    "filter_cutoff": 6.0,
    "min_frames_required": 15,
    "force_reprocess": True,  # <--- FORCES AI TO READ EVERY FRAME OF THE NEW VIDEO
    "target_fps": 60.0 
}

DET_MODEL = 'https://huggingface.co/datasets/DavidPagnon/rtmlib_models/resolve/main/mmpose/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx'
POSE_MODEL = 'https://huggingface.co/datasets/DavidPagnon/rtmlib_models/resolve/main/mmpose/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.onnx'

class CustomBody(Body):
    def __init__(self, *args, **kwargs):
        super(CustomBody, self).__init__(
            det=DET_MODEL, 
            det_input_size=(416, 416), 
            pose=POSE_MODEL, 
            pose_input_size=(192, 256), 
            backend='onnxruntime', 
            device='cpu'
        )

def calculate_velocity_graph_angle(velocity_data):
    y_range = np.max(velocity_data) - np.min(velocity_data)
    x_range = len(velocity_data)
    
    if y_range == 0: y_range = 1 
    
    dy = np.diff(velocity_data)
    dy_percent = dy / y_range
    dx_percent = 1.0 / x_range
    
    visual_ratio = 4.0 / 11.0
    
    return np.degrees(np.arctan2(dy_percent * visual_ratio, dx_percent))

def smooth_1d(data, window_size=5):
    if len(data) < window_size: return data
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    pad = window_size // 2
    smoothed[:pad] = data[:pad]
    smoothed[-pad:] = data[-pad:]
    return smoothed

def lowpass_filter_coords(coords, cutoff, fs, order=2):
    if len(coords) < 15: 
        return coords
    b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    return np.column_stack((filtfilt(b, a, coords[:, 0]), filtfilt(b, a, coords[:, 1])))

def find_heel_strikes(y_coords, min_dist=45, prominence_factor=0.12):
    peaks = []
    y_range = np.max(y_coords) - np.min(y_coords)
    if y_range <= 0: return np.array([])
    threshold = np.min(y_coords) + 0.35 * y_range
    for i in range(1, len(y_coords) - 1):
        if y_coords[i] > y_coords[i-1] and y_coords[i] > y_coords[i+1] and y_coords[i] > threshold:
            left_slice = y_coords[max(0, i-min_dist):i]
            prom_l = y_coords[i] - np.min(left_slice) if len(left_slice) > 0 else 0
            if prom_l > y_range * prominence_factor:
                peaks.append(i)
    resolved = []
    if peaks:
        for p in sorted(peaks):
            if not resolved or p - resolved[-1] >= min_dist:
                resolved.append(p)
            elif y_coords[p] > y_coords[resolved[-1]]:
                resolved[-1] = p
    return np.array(resolved)

def find_toe_offs(toe_y_coords, initial_contacts):
    toe_offs = []
    for ic_idx in initial_contacts:
        baseline_y = toe_y_coords[ic_idx, 1]
        for i in range(ic_idx, len(toe_y_coords)):
            if toe_y_coords[i, 1] < baseline_y - 2.0:
                toe_offs.append(i)
                break
    return np.array(toe_offs)

def main():
    csv_path = CONFIG.get("data_save_path", "gait_data.csv")
    fps = CONFIG.get("target_fps", 60.0) 
    
    if os.path.exists(csv_path) and not CONFIG.get("force_reprocess", True):
        print("Loading existing CSV data...")
        df = pd.read_csv(csv_path)
        
        timestamps = np.arange(len(df)) * (1.0 / fps)
        
        r_heel = df[['r_heel_x', 'r_heel_y']].values
        l_heel = df[['l_heel_x', 'l_heel_y']].values
        r_toe = df[['r_toe_x', 'r_toe_y']].values
        l_toe = df[['l_toe_x', 'l_toe_y']].values

    else:
        print("Processing raw video with AI tracking...")
        pose_model = PoseTracker(solution=CustomBody, det_frequency=10, tracking=False, backend='onnxruntime', device='cpu')
        video = cv2.VideoCapture(CONFIG["video_path"])
        
        if not video.isOpened(): return

        dt = 1.0 / fps

        r_heel, l_heel, r_toe, l_toe, timestamps = [], [], [], [], []
        frame_idx = 0
        prev_time = time.time()

        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            keypoints, scores = pose_model(frame)
            timestamps.append(frame_idx * dt)

            if len(keypoints) > 0:
                kpts = keypoints[0]
                r_heel.append(kpts[25]); l_heel.append(kpts[24])
                r_toe.append(kpts[21]);  l_toe.append(kpts[20])
            else:
                for arr in (r_heel, l_heel, r_toe, l_toe):
                    arr.append(arr[-1] if arr else np.array([0.0, 0.0]))

            canvas = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)
            
            curr_time = time.time()
            process_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            print(f"\rProcessing Video... FPS: {process_fps:.1f}   ", end="", flush=True)
            
            frame_idx += 1
            
            cv2.imshow('Pose Estimation', canvas)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'): 
                break

        print()
        video.release()
        cv2.destroyAllWindows()

        r_heel, l_heel = np.array(r_heel), np.array(l_heel)
        r_toe, l_toe = np.array(r_toe), np.array(l_toe)
        timestamps = np.array(timestamps)

        df = pd.DataFrame({
            'time': timestamps,
            'r_heel_x': r_heel[:, 0], 'r_heel_y': r_heel[:, 1],
            'l_heel_x': l_heel[:, 0], 'l_heel_y': l_heel[:, 1],
            'r_toe_x': r_toe[:, 0],  'r_toe_y': r_toe[:, 1],
            'l_toe_x': l_toe[:, 0],  'l_toe_y': l_toe[:, 1],
        })
        df.to_csv(csv_path, index=False)

    if len(timestamps) < CONFIG["min_frames_required"]: return
    dt = 1.0 / fps

    print(f"\n--- Processing Full Segment (Total: {len(timestamps)} frames) ---")

    sr_heel = lowpass_filter_coords(r_heel, CONFIG["filter_cutoff"], fps)
    sl_heel = lowpass_filter_coords(l_heel, CONFIG["filter_cutoff"], fps)
    sr_toe = lowpass_filter_coords(r_toe, CONFIG["filter_cutoff"], fps)
    sl_toe = lowpass_filter_coords(l_toe, CONFIG["filter_cutoff"], fps)

    vr_heel = smooth_1d(np.sqrt(np.diff(sr_heel[:, 0])**2 + np.diff(sr_heel[:, 1])**2) / dt)
    vl_heel = smooth_1d(np.sqrt(np.diff(sl_heel[:, 0])**2 + np.diff(sl_heel[:, 1])**2) / dt)
    ar_heel = smooth_1d(np.diff(vr_heel) / dt)
    al_heel = smooth_1d(np.diff(vl_heel) / dt)

    print(f"\n--- Velocity Array Lengths: Right={len(vr_heel)}, Left={len(vl_heel)} ---")

    angle_vr_heel = calculate_velocity_graph_angle(vr_heel)
    angle_vl_heel = calculate_velocity_graph_angle(vl_heel)

    print(f"\n--- Heel Velocity Graph Angles (Full Data: {len(angle_vr_heel)} Segments) ---")
    for i in range(min(len(angle_vr_heel), len(angle_vl_heel))):
        print(f"Frames ({i}-{i+1}) | Right: {angle_vr_heel[i]:7.2f}° | Left: {angle_vl_heel[i]:7.2f}°")

    ric = find_heel_strikes(sr_heel[:, 1])
    lic = find_heel_strikes(sl_heel[:, 1])
    rto = find_toe_offs(sr_toe, lic) 
    lto = find_toe_offs(sl_toe, ric)

    fig, axs = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    
    axs[0].plot(timestamps, sr_heel[:, 1], label='Right Heel', color='#1f77b4', linewidth=2)
    axs[0].plot(timestamps, sl_heel[:, 1], label='Left Heel', color='#d62728', linewidth=2)
    axs[0].invert_yaxis()
    axs[0].set_title('Vertical Joint Trajectories')
    axs[0].set_ylabel('Position (Pixels)')
    
    axs[1].plot(timestamps[1:], vr_heel, label='Right Vel', color='#1f77b4', linewidth=2)
    axs[1].plot(timestamps[1:], vl_heel, label='Left Vel', color='#d62728', linewidth=2)
    axs[1].set_title('Heel Velocity Magnitude')
    axs[1].set_ylabel('Velocity (px/s)')
    
    axs[2].plot(timestamps[2:], ar_heel, label='Right Accel', color='#1f77b4', linewidth=2)
    axs[2].plot(timestamps[2:], al_heel, label='Left Accel', color='#d62728', linewidth=2)
    axs[2].set_title('Heel Acceleration Magnitude')
    axs[2].set_ylabel('Acceleration (px/s²)')
    axs[2].set_xlabel('Time (seconds)')

    for ic in ric: axs[0].axvline(timestamps[ic], color='#1f77b4', linestyle=':', label='R Strike' if ic==ric[0] else "")
    for ic in lic: axs[0].axvline(timestamps[ic], color='#d62728', linestyle=':', label='L Strike' if ic==lic[0] else "")

    for ax in axs: 
        ax.grid(True, alpha=0.5)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()