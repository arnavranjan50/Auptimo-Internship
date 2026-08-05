import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import savgol_filter
from rtmlib.tools.solution.pose_tracker import PoseTracker
from rtmlib.tools.solution.body import Body

CONFIG = {
    "video_path": r"C:\Users\Arnav Ranjan\OneDrive\Desktop\GaitON Internship\WhatsApp Video 2026-06-16 at 19.13.40.mp4",
    "data_save_path": r"C:\Users\Arnav Ranjan\OneDrive\Desktop\GaitON Internship\gait_data.csv",
    "savgol_window": 11,
    "savgol_polyorder": 3,
    "min_frames_required": 15
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

def smooth_1d(data, window_size=5):
    if len(data) < window_size: return data
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    pad = window_size // 2
    smoothed[:pad] = data[:pad]
    smoothed[-pad:] = data[-pad:]
    return smoothed

def filter_coords_savgol(coords, window_length=11, polyorder=3):
    if len(coords) < window_length: 
        return coords
    if window_length % 2 == 0:
        window_length += 1
    x_filtered = savgol_filter(coords[:, 0], window_length, polyorder)
    y_filtered = savgol_filter(coords[:, 1], window_length, polyorder)
    return np.column_stack((x_filtered, y_filtered))

def isolate_walking_segment(r_heel, l_heel, timestamps):
    fps = 30.0
    if len(timestamps) > 1:
        fps = 1.0 / (timestamps[1] - timestamps[0])
    window_size = int(51 * (fps / 60.0))
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(5, window_size)

    avg_x = (r_heel[:, 0] + l_heel[:, 0]) / 2.0
    avg_x_smooth = smooth_1d(avg_x, window_size=window_size) if len(avg_x) >= window_size else avg_x
    
    dx = np.zeros_like(avg_x_smooth)
    dx[1:] = np.diff(avg_x_smooth)
    
    ltr_mask = (dx * fps) > 15.0
    
    segments, start = [], None
    min_len = int(30 * (fps / 60.0))
    min_len = max(5, min_len)
    
    for i, is_ltr in enumerate(ltr_mask):
        if is_ltr and start is None: start = i
        elif not is_ltr and start is not None:
            if i - start >= min_len: segments.append((start, i))
            start = None
    if start is not None and len(ltr_mask) - start >= min_len:
        segments.append((start, len(ltr_mask)))
    if not segments: return 0, len(timestamps)
    return max(segments, key=lambda s: s[1] - s[0])

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

def calculate_velocity_vector_angle(coords):
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    return np.degrees(np.arctan2(-dy, dx))

def calculate_velocity_graph_angle(velocity_data):
    y_range = np.max(velocity_data) - np.min(velocity_data)
    x_range = len(velocity_data)
    if y_range == 0: y_range = 1 
    
    dy = np.diff(velocity_data)
    dy_percent = dy / y_range
    dx_percent = 1.0 / x_range
    visual_ratio = 4.0 / 11.0
    
    return np.degrees(np.arctan2(dy_percent * visual_ratio, dx_percent))

def main():
    csv_path = CONFIG.get("data_save_path", "gait_data.csv")
    fps = 30.0 
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        timestamps = df['time'].values
        r_heel = df[['r_heel_x', 'r_heel_y']].values
        l_heel = df[['l_heel_x', 'l_heel_y']].values
        r_toe = df[['r_toe_x', 'r_toe_y']].values
        l_toe = df[['l_toe_x', 'l_toe_y']].values
        if len(timestamps) > 1:
            fps = 1.0 / (timestamps[1] - timestamps[0])

    else:
        pose_model = PoseTracker(solution=CustomBody, det_frequency=10, tracking=False, backend='onnxruntime', device='cpu')
        video = cv2.VideoCapture(CONFIG["video_path"])
        
        if not video.isOpened(): return

        fps = 60.0
        dt = 1.0 / fps

        r_heel, l_heel, r_toe, l_toe, timestamps = [], [], [], [], []
        frame_idx = 0

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

            frame_idx += 1

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

    start_idx, end_idx = isolate_walking_segment(r_heel, l_heel, timestamps)

    sr_heel = filter_coords_savgol(r_heel, CONFIG["savgol_window"], CONFIG["savgol_polyorder"])
    sl_heel = filter_coords_savgol(l_heel, CONFIG["savgol_window"], CONFIG["savgol_polyorder"])
    sr_toe = filter_coords_savgol(r_toe, CONFIG["savgol_window"], CONFIG["savgol_polyorder"])
    sl_toe = filter_coords_savgol(l_toe, CONFIG["savgol_window"], CONFIG["savgol_polyorder"])

    vr_heel = smooth_1d(np.sqrt(np.diff(sr_heel[:, 0])**2 + np.diff(sr_heel[:, 1])**2) / dt)
    vl_heel = smooth_1d(np.sqrt(np.diff(sl_heel[:, 0])**2 + np.diff(sl_heel[:, 1])**2) / dt)
    ar_heel = smooth_1d(np.diff(vr_heel) / dt)
    al_heel = smooth_1d(np.diff(vl_heel) / dt)

    vector_angle_r_heel = calculate_velocity_vector_angle(sr_heel)
    vector_angle_l_heel = calculate_velocity_vector_angle(sl_heel)

    graph_angle_r_heel = calculate_velocity_graph_angle(vr_heel)
    graph_angle_l_heel = calculate_velocity_graph_angle(vl_heel)

    print(f"\n--- Heel Velocity Vector Angles (Full: {len(vector_angle_r_heel)} Transitions, Printed up to Frame 912) ---")
    for i in range(len(vector_angle_r_heel)):
        if i >= 912:
            break
        print(f"Frames ({i}-{i+1}) | Right Vector Angle: {vector_angle_r_heel[i]:7.2f}° | Left Vector Angle: {vector_angle_l_heel[i]:7.2f}°")

    print(f"\n--- Heel Velocity Graph Slope Angles (Full: {len(graph_angle_r_heel)} Transitions, Printed up to Frame 912) ---")
    for i in range(len(graph_angle_r_heel)):
        if i >= 912:
            break
        print(f"Frames ({i}-{i+1}) | Right Graph Slope Angle: {graph_angle_r_heel[i]:7.2f}° | Left Graph Slope Angle: {graph_angle_l_heel[i]:7.2f}°")

    ric = find_heel_strikes(sr_heel[:, 1])
    lic = find_heel_strikes(sl_heel[:, 1])
    rto = find_toe_offs(sr_toe, lic) 
    lto = find_toe_offs(sl_toe, ric)

    fig, axs = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    
    axs[0].plot(timestamps[:912], sr_heel[:912, 1], label='Right Heel', color='#1f77b4', linewidth=2)
    axs[0].plot(timestamps[:912], sl_heel[:912, 1], label='Left Heel', color='#d62728', linewidth=2)
    axs[0].invert_yaxis()
    axs[0].set_title('Vertical Joint Trajectories')
    axs[0].set_ylabel('Position (Pixels)')
    
    axs[1].plot(timestamps[1:912], vr_heel[:911], label='Right Vel', color='#1f77b4', linewidth=2)
    axs[1].plot(timestamps[1:912], vl_heel[:911], label='Left Vel', color='#d62728', linewidth=2)
    axs[1].set_title('Heel Velocity Magnitude')
    axs[1].set_ylabel('Velocity (px/s)')
    
    axs[2].plot(timestamps[2:912], ar_heel[:910], label='Right Accel', color='#1f77b4', linewidth=2)
    axs[2].plot(timestamps[2:912], al_heel[:910], label='Left Accel', color='#d62728', linewidth=2)
    axs[2].set_title('Heel Acceleration Magnitude')
    axs[2].set_ylabel('Acceleration (px/s²)')
    axs[2].set_xlabel('Time (seconds)')

    first_r_strike = next((ic for ic in ric if ic < 912), None)
    first_l_strike = next((ic for ic in lic if ic < 912), None)
    for ic in ric:
        if ic < 912:
            axs[0].axvline(timestamps[ic], color='#1f77b4', linestyle=':', label='R Strike' if ic==first_r_strike else "")
    for ic in lic:
        if ic < 912:
            axs[0].axvline(timestamps[ic], color='#d62728', linestyle=':', label='L Strike' if ic==first_l_strike else "")

    for ax in axs: 
        ax.grid(True, alpha=0.5)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(r"C:\Users\Arnav Ranjan\.gemini\antigravity-ide\brain\53d74ca4-784f-4c10-abe1-f3eda2dcf83a\kinematics_plot.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()