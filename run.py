import cv2
import numpy as np
import matplotlib.pyplot as plt
from rtmlib import Body, PoseTracker, draw_skeleton
from scipy.signal import butter, filtfilt

class CustomBody(Body):
    def __init__(self, *args, **kwargs):
        super().__init__(
            det='https://huggingface.co/datasets/DavidPagnon/rtmlib_models/resolve/main/mmpose/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx',
            det_input_size=(416, 416),
            pose='https://huggingface.co/datasets/DavidPagnon/rtmlib_models/resolve/main/mmpose/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.onnx',
            pose_input_size=(192, 256),
            backend='onnxruntime',
            device='cpu'
        )

pose_model = PoseTracker(
    solution=CustomBody,
    det_frequency=10,
    tracking=False,
    backend='onnxruntime',
    device='cpu'
)

video_path = r'C:\Users\Arnav Ranjan\OneDrive\Desktop\GaitON Internship\WhatsApp Video 2026-06-16 at 19.13.40.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0 
dt = 1.0 / fps

delay = max(1, int(1000 / fps))

right_heel_positions = []
left_heel_positions = []
right_toe_positions = []
left_toe_positions = []
timestamps = []

print("Starting video processing. Press 'q' in the window to quit.")
frame_idx = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    keypoints, scores = pose_model(frame)

    t = frame_idx * dt
    timestamps.append(t)

    if len(keypoints) > 0:
        person_kpts = keypoints[0]
        right_heel_positions.append(person_kpts[25])  
        left_heel_positions.append(person_kpts[24])   
        right_toe_positions.append(person_kpts[21])   
        left_toe_positions.append(person_kpts[20])    
    else:

        right_heel_positions.append(right_heel_positions[-1] if right_heel_positions else np.array([0.0, 0.0]))
        left_heel_positions.append(left_heel_positions[-1] if left_heel_positions else np.array([0.0, 0.0]))
        right_toe_positions.append(right_toe_positions[-1] if right_toe_positions else np.array([0.0, 0.0]))
        left_toe_positions.append(left_toe_positions[-1] if left_toe_positions else np.array([0.0, 0.0]))

    canvas = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)
    cv2.imshow('Pose Estimation', canvas)

    frame_idx += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

r_heel = np.array(right_heel_positions)
l_heel = np.array(left_heel_positions)
r_toe = np.array(right_toe_positions)
l_toe = np.array(left_toe_positions)
timestamps = np.array(timestamps)

total_frames = len(timestamps)
if total_frames < 15:
    print("Not enough frames processed to perform gait analysis (minimum 15 required).")
    exit(0)

avg_x = (r_heel[:, 0] + l_heel[:, 0]) / 2.0

w_dir = 51
if len(avg_x) >= w_dir:
    avg_x_smooth = np.convolve(avg_x, np.ones(w_dir)/w_dir, mode='same')

    pad_dir = w_dir // 2
    avg_x_smooth[:pad_dir] = avg_x[:pad_dir]
    avg_x_smooth[-pad_dir:] = avg_x[-pad_dir:]
else:
    avg_x_smooth = avg_x

dx = np.zeros_like(avg_x_smooth)
dx[1:] = np.diff(avg_x_smooth)

ltr_mask = dx > 0.5
segments = []
start = None
min_segment_len = 30 
for i in range(len(ltr_mask)):
    if ltr_mask[i] and start is None:
        start = i
    elif not ltr_mask[i] and start is not None:
        if i - start >= min_segment_len:
            segments.append((start, i))
        start = None
if start is not None and len(ltr_mask) - start >= min_segment_len:
    segments.append((start, len(ltr_mask)))

if not segments:
    start_idx = 0
    end_idx = len(timestamps)
    print("\nWarning: No distinct Left-to-Right walking segment detected. Analyzing full video.")
else:
    longest_seg = max(segments, key=lambda s: s[1] - s[0])
    start_idx, end_idx = longest_seg
    print(f"\nIsolated primary Left-to-Right walking segment: Frame {start_idx} to {end_idx} "
          f"({start_idx*dt:.2f}s to {end_idx*dt:.2f}s)")

r_heel_cropped = r_heel[start_idx:end_idx]
l_heel_cropped = l_heel[start_idx:end_idx]
r_toe_cropped = r_toe[start_idx:end_idx]
l_toe_cropped = l_toe[start_idx:end_idx]
timestamps_cropped = timestamps[start_idx:end_idx]

cropped_frames = len(timestamps_cropped)
if cropped_frames < 15:
    print("Cropped left-to-right segment is too short for analysis (minimum 15 frames).")
    exit(0)

def smooth_1d(data, w=5):
    if len(data) < w:
        return data
    smoothed = np.convolve(data, np.ones(w)/w, mode='same')
    pad = w // 2
    smoothed[:pad] = data[:pad]
    smoothed[-pad:] = data[-pad:]
    return smoothed

def lowpass_filter_coords(coords, cutoff=6.0, fs=100.0, order=2):
    if len(coords) < 15:
        return coords
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    sx = filtfilt(b, a, coords[:, 0])
    sy = filtfilt(b, a, coords[:, 1])
    return np.column_stack((sx, sy))

smooth_r_heel = lowpass_filter_coords(r_heel_cropped, cutoff=6.0, fs=fps)
smooth_l_heel = lowpass_filter_coords(l_heel_cropped, cutoff=6.0, fs=fps)
smooth_r_toe = lowpass_filter_coords(r_toe_cropped, cutoff=6.0, fs=fps)
smooth_l_toe = lowpass_filter_coords(l_toe_cropped, cutoff=6.0, fs=fps)

r_heel_dx = np.diff(smooth_r_heel[:, 0])
r_heel_dy = np.diff(smooth_r_heel[:, 1])
r_heel_dist = np.sqrt(r_heel_dx**2 + r_heel_dy**2)

l_heel_dx = np.diff(smooth_l_heel[:, 0])
l_heel_dy = np.diff(smooth_l_heel[:, 1])
l_heel_dist = np.sqrt(l_heel_dx**2 + l_heel_dy**2)

v_r_heel_mag = r_heel_dist / dt
v_l_heel_mag = l_heel_dist / dt

v_r_heel_mag_smooth = smooth_1d(v_r_heel_mag, w=5)
v_l_heel_mag_smooth = smooth_1d(v_l_heel_mag, w=5)

a_r_heel_mag = np.diff(v_r_heel_mag_smooth) / dt
a_l_heel_mag = np.diff(v_l_heel_mag_smooth) / dt

a_r_heel_mag_smooth = smooth_1d(a_r_heel_mag, w=5)
a_l_heel_mag_smooth = smooth_1d(a_l_heel_mag, w=5)

def find_heel_strikes(y, min_dist=45, prominence_factor=0.12):
    peaks = []
    n = len(y)
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    if y_range <= 0:
        return np.array([])
    threshold = y_min + 0.35 * y_range

    for i in range(1, n - 1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            if y[i] > threshold:
                left_slice = y[max(0, i-min_dist):i]
                prom_l = y[i] - np.min(left_slice) if len(left_slice) > 0 else 0
                if prom_l > y_range * prominence_factor:
                    peaks.append(i)

    resolved = []
    if peaks:
        peaks = sorted(peaks)
        resolved.append(peaks[0])
        for p in peaks[1:]:
            if p - resolved[-1] < min_dist:
                if y[p] > y[resolved[-1]]:
                    resolved[-1] = p
            else:
                resolved.append(p)
    return np.array(resolved)

ric_frames_rel = find_heel_strikes(smooth_r_heel[:, 1], min_dist=45, prominence_factor=0.12)
lic_frames_rel = find_heel_strikes(smooth_l_heel[:, 1], min_dist=45, prominence_factor=0.12)

rto_frames_rel = []
for lic_rel_idx in lic_frames_rel:

    baseline_y = smooth_r_toe[lic_rel_idx, 1]

    found = None
    for i in range(lic_rel_idx, len(smooth_r_toe)):
        if smooth_r_toe[i, 1] < baseline_y - 2.0:
            found = i
            break
    if found is not None:
        rto_frames_rel.append(found)

lto_frames_rel = []
for ric_rel_idx in ric_frames_rel:

    baseline_y = smooth_l_toe[ric_rel_idx, 1]

    found = None
    for i in range(ric_rel_idx, len(smooth_l_toe)):
        if smooth_l_toe[i, 1] < baseline_y - 2.0:
            found = i
            break
    if found is not None:
        lto_frames_rel.append(found)

rto_frames_rel = np.array(rto_frames_rel)
lto_frames_rel = np.array(lto_frames_rel)

ric_frames = ric_frames_rel + start_idx
lic_frames = lic_frames_rel + start_idx
rto_frames = rto_frames_rel + start_idx
lto_frames = lto_frames_rel + start_idx

r_lr_phases = []
for ric in ric_frames:
    post_lto = [lto for lto in lto_frames if lto > ric]
    if post_lto:
        lto = post_lto[0]
        if (lto - ric) * dt < 0.45:
            r_lr_phases.append((ric, lto))

l_lr_phases = []
for lic in lic_frames:
    post_rto = [rto for rto in rto_frames if rto > lic]
    if post_rto:
        rto = post_rto[0]
        if (rto - lic) * dt < 0.45:
            l_lr_phases.append((lic, rto))

border = 20
v_r_heel_mag_smooth_mid = v_r_heel_mag_smooth[border:-border] if len(v_r_heel_mag_smooth) > 2 * border else v_r_heel_mag_smooth
v_l_heel_mag_smooth_mid = v_l_heel_mag_smooth[border:-border] if len(v_l_heel_mag_smooth) > 2 * border else v_l_heel_mag_smooth
a_r_heel_mag_smooth_mid = a_r_heel_mag_smooth[border:-border] if len(a_r_heel_mag_smooth) > 2 * border else a_r_heel_mag_smooth
a_l_heel_mag_smooth_mid = a_l_heel_mag_smooth[border:-border] if len(a_l_heel_mag_smooth) > 2 * border else a_l_heel_mag_smooth

print("\n" + "="*70)
print("             GAIT ANALYSIS REPORT (LEFT-TO-RIGHT DIRECTION)")
print("="*70)
print(f"Video File       : {video_path}")
print(f"Frame Rate       : {fps:.2f} FPS")
print(f"Segment Analyzed : Frame {start_idx} to {end_idx} (Duration: {cropped_frames * dt:.2f}s)")
print("-"*70)

print("INITIAL CONTACTS (HEEL STRIKES) TIMELINE:")
all_contacts = []
for ric in ric_frames:
    all_contacts.append((ric, "Right"))
for lic in lic_frames:
    all_contacts.append((lic, "Left"))
all_contacts = sorted(all_contacts, key=lambda x: x[0])

if all_contacts:
    for frame, side in all_contacts:
        print(f"  - Frame {frame:4d} | Time {frame*dt:5.2f}s | {side} Initial Contact")
else:
    print("  No Initial Contacts detected in this segment.")
print("-"*70)

print("LOADING RESPONSE PHASES:")
has_lr = False
if r_lr_phases:
    has_lr = True
    print("  [Right Loading Response - RIC to LTO]")
    for i, (ric, lto) in enumerate(r_lr_phases, 1):
        dur = (lto - ric) * dt
        print(f"    Phase {i}: Frame {ric:4d} -> {lto:4d} | Time {ric*dt:5.2f}s -> {lto*dt:5.2f}s | Duration: {dur:.3f}s")

if l_lr_phases:
    has_lr = True
    print("  [Left Loading Response - LIC to RTO]")
    for i, (lic, rto) in enumerate(l_lr_phases, 1):
        dur = (rto - lic) * dt
        print(f"    Phase {i}: Frame {lic:4d} -> {rto:4d} | Time {lic*dt:5.2f}s -> {rto*dt:5.2f}s | Duration: {dur:.3f}s")

if not has_lr:
    print("  No Loading Response phases detected in this segment.")
print("-"*70)

print("HEEL KINEMATICS SUMMARY (LEFT-TO-RIGHT, EXCLUDING BOUNDARY FRAMES):")
print(f"  - Avg Right Heel Velocity: {np.mean(v_r_heel_mag_smooth_mid):.2f} pixels/sec")
print(f"  - Avg Left Heel Velocity : {np.mean(v_l_heel_mag_smooth_mid):.2f} pixels/sec")
print(f"  - Max Right Heel Accel   : {np.max(a_r_heel_mag_smooth_mid):.2f} pixels/sec²")
print(f"  - Max Left Heel Accel    : {np.max(a_l_heel_mag_smooth_mid):.2f} pixels/sec²")
print("="*70 + "\n")

fig, axs = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

axs[0].plot(timestamps_cropped, smooth_r_heel[:, 1], label='Right Heel Y', color='#1f77b4', linewidth=1.8)
axs[0].plot(timestamps_cropped, smooth_l_heel[:, 1], label='Left Heel Y', color='#d62728', linewidth=1.8)
axs[0].plot(timestamps_cropped, smooth_r_toe[:, 1], label='Right Toe Y', color='#9ecae1', linestyle='--', alpha=0.8)
axs[0].plot(timestamps_cropped, smooth_l_toe[:, 1], label='Left Toe Y', color='#ff9896', linestyle='--', alpha=0.8)
axs[0].set_ylabel('Vertical Position (pixels, lower is ground)')
axs[0].set_title('Joint Vertical Trajectories & Gait Phase Detections')
axs[0].invert_yaxis()
axs[0].legend(loc='upper right')
axs[0].grid(True, linestyle='--', alpha=0.5)

axs[1].plot(timestamps_cropped[1:], v_r_heel_mag_smooth, label='Right Heel Velocity Magnitude', color='#1f77b4', linewidth=2)
axs[1].plot(timestamps_cropped[1:], v_l_heel_mag_smooth, label='Left Heel Velocity Magnitude', color='#d62728', linewidth=2)
axs[1].set_ylabel('Velocity (pixels/sec)')
axs[1].set_title('Heel Velocity Magnitude')
axs[1].legend(loc='upper right')
axs[1].grid(True, linestyle='--', alpha=0.5)

v_max_y = max(np.percentile(v_r_heel_mag_smooth_mid, 99), np.percentile(v_l_heel_mag_smooth_mid, 99)) * 1.3
axs[1].set_ylim(0, v_max_y)

axs[2].plot(timestamps_cropped[2:], a_r_heel_mag_smooth, label='Right Heel Acceleration Magnitude', color='#1f77b4', linewidth=2)
axs[2].plot(timestamps_cropped[2:], a_l_heel_mag_smooth, label='Left Heel Acceleration Magnitude', color='#d62728', linewidth=2)
axs[2].set_ylabel('Acceleration (pixels/sec²)')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_title('Heel Acceleration Magnitude')
axs[2].legend(loc='upper right')
axs[2].grid(True, linestyle='--', alpha=0.5)

a_max_y = max(np.percentile(a_r_heel_mag_smooth_mid, 99), np.percentile(a_l_heel_mag_smooth_mid, 99)) * 1.3
a_min_y = min(np.percentile(a_r_heel_mag_smooth_mid, 1), np.percentile(a_l_heel_mag_smooth_mid, 1)) * 1.3
axs[2].set_ylim(a_min_y, a_max_y)

label_ric = 'RIC (Heel Strike)'
label_lic = 'LIC (Heel Strike)'
label_rlr = 'Right Loading Response'
label_llr = 'Left Loading Response'

for ric in ric_frames:
    t_ric = timestamps[ric]
    for ax in axs:
        ax.axvline(x=t_ric, color='#1f77b4', linestyle=':', alpha=0.75, label=label_ric)
    label_ric = None

for lic in lic_frames:
    t_lic = timestamps[lic]
    for ax in axs:
        ax.axvline(x=t_lic, color='#d62728', linestyle=':', alpha=0.75, label=label_lic)
    label_lic = None

for ric, lto in r_lr_phases:
    t_start = timestamps[ric]
    t_end = timestamps[lto]
    for ax in axs:
        ax.axvspan(t_start, t_end, color='#1f77b4', alpha=0.12, label=label_rlr)
    label_rlr = None

for lic, rto in l_lr_phases:
    t_start = timestamps[lic]
    t_end = timestamps[rto]
    for ax in axs:
        ax.axvspan(t_start, t_end, color='#d62728', alpha=0.12, label=label_llr)
    label_llr = None

for ax in axs:
    handles, labels = ax.get_legend_handles_labels()

    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.savefig(r'C:\Users\Arnav Ranjan\.gemini\antigravity-ide\brain\c7e04f53-a826-4e08-8546-08e56cb30662\kinematics_plot.png', dpi=300)
plt.show()
