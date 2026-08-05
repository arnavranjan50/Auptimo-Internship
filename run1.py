import cv2
import numpy as np
import matplotlib.pyplot as plt
from rtmlib import Body, PoseTracker, draw_skeleton

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
    print("Error: Could not open video.")
    exit(1)

fps = video.get(cv2.CAP_PROP_FPS) or 30.0   
dt  = 1.0 / fps                              
delay = max(1, int(1000 / fps))              

r_heel_list, l_heel_list = [], []
r_toe_list,  l_toe_list  = [], []
timestamps = []

print("Processing video… Press 'q' to stop early.")
frame_idx = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    keypoints, scores = pose_model(frame)
    timestamps.append(frame_idx * dt)

    if len(keypoints) > 0:
        kpts = keypoints[0]            
        r_heel_list.append(kpts[25])
        l_heel_list.append(kpts[24])
        r_toe_list.append(kpts[21])
        l_toe_list.append(kpts[20])
    else:

        fallback = lambda lst: lst[-1] if lst else np.array([0.0, 0.0])
        r_heel_list.append(fallback(r_heel_list))
        l_heel_list.append(fallback(l_heel_list))
        r_toe_list.append(fallback(r_toe_list))
        l_toe_list.append(fallback(l_toe_list))

    canvas = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)
    cv2.imshow('Pose Estimation', canvas)
    frame_idx += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f"Done. {frame_idx} frames processed.")

r_heel = np.array(r_heel_list)    
l_heel = np.array(l_heel_list)
r_toe  = np.array(r_toe_list)
l_toe  = np.array(l_toe_list)
timestamps = np.array(timestamps)
N = len(timestamps)

if N < 15:
    print("Too few frames — cannot analyse.")
    exit(0)

def moving_avg(data, w=7):
    kernel = np.ones(w) / w
    if data.ndim == 1:
        return np.convolve(data, kernel, mode='same')

    return np.column_stack(
        [np.convolve(data[:, col], kernel, mode='same') for col in range(data.shape[1])]
    )

W = 7   
sm_r_heel = moving_avg(r_heel, W)
sm_l_heel = moving_avg(l_heel, W)
sm_r_toe  = moving_avg(r_toe,  W)
sm_l_toe  = moving_avg(l_toe,  W)

def kinematics(coords, dt):
    dx = np.diff(coords[:, 0])              
    dy = np.diff(coords[:, 1])              
    speed = np.sqrt(dx**2 + dy**2) / dt    
    speed = moving_avg(speed, 5)            

    accel = np.diff(speed) / dt             
    accel = moving_avg(accel, 5)

    return speed, accel

v_r, a_r = kinematics(sm_r_heel, dt)
v_l, a_l = kinematics(sm_l_heel, dt)

t_v = timestamps[1:]     
t_a = timestamps[2:]     

def find_peaks(y, min_gap=30, threshold_pct=0.35):
    threshold = y.min() + threshold_pct * (y.max() - y.min())
    peaks = []

    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and y[i] > threshold:
            peaks.append(i)

    merged = []
    if peaks:
        merged.append(peaks[0])
        for p in peaks[1:]:
            if p - merged[-1] < min_gap:
                merged[-1] = p if y[p] > y[merged[-1]] else merged[-1]
            else:
                merged.append(p)

    return np.array(merged)

ric = find_peaks(sm_r_heel[:, 1])    
lic = find_peaks(sm_l_heel[:, 1])    

border = 10
v_r_mid = v_r[border:-border] if len(v_r) > 2 * border else v_r
v_l_mid = v_l[border:-border] if len(v_l) > 2 * border else v_l
a_r_mid = a_r[border:-border] if len(a_r) > 2 * border else a_r
a_l_mid = a_l[border:-border] if len(a_l) > 2 * border else a_l

print("\n" + "=" * 60)
print("        GAIT ANALYSIS REPORT")
print("=" * 60)
print(f"FPS: {fps:.1f}   |   Frames: {N}   |   Duration: {N * dt:.2f}s")
print("-" * 60)

print("HEEL STRIKES (INITIAL CONTACTS):")
all_ic = sorted([(f, 'Right') for f in ric] + [(f, 'Left') for f in lic])
if all_ic:
    for frame, side in all_ic:
        print(f"  Frame {frame:4d}  |  {frame * dt:.2f}s  |  {side}")
else:
    print("  None detected.")
print("-" * 60)

print("KINEMATICS SUMMARY (boundary frames excluded):")
print(f"  Avg Right Heel Speed  : {np.mean(v_r_mid):.1f}  px/s")
print(f"  Avg Left  Heel Speed  : {np.mean(v_l_mid):.1f}  px/s")
print(f"  Peak Right Heel Accel : {np.max(a_r_mid):.1f}  px/s²")
print(f"  Peak Left  Heel Accel : {np.max(a_l_mid):.1f}  px/s²")
print("=" * 60 + "\n")

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Gait Analysis — Heel Kinematics", fontsize=14, fontweight='bold')

BLUE   = 'steelblue'
RED    = 'crimson'

axs[0].plot(timestamps, sm_r_heel[:, 1], color=BLUE, label='Right Heel Y', lw=1.8)
axs[0].plot(timestamps, sm_l_heel[:, 1], color=RED,  label='Left  Heel Y', lw=1.8)
axs[0].invert_yaxis()    
axs[0].set_ylabel('Vertical Position (px)')
axs[0].set_title('Heel Vertical Position  (Y-axis flipped: lower = ground)')
axs[0].legend(loc='upper right')
axs[0].grid(alpha=0.35)

for f in ric:
    axs[0].axvline(timestamps[f], color=BLUE, linestyle=':', alpha=0.75, lw=1.2)
for f in lic:
    axs[0].axvline(timestamps[f], color=RED,  linestyle=':', alpha=0.75, lw=1.2)

axs[1].plot(t_v, v_r, color=BLUE, label='Right Heel Speed', lw=1.8)
axs[1].plot(t_v, v_l, color=RED,  label='Left  Heel Speed', lw=1.8)
axs[1].set_ylabel('Speed (px/s)')
axs[1].set_title('Heel Speed Magnitude')
axs[1].legend(loc='upper right')
axs[1].grid(alpha=0.35)

v_ceil = max(np.percentile(v_r_mid, 99), np.percentile(v_l_mid, 99)) * 1.3
axs[1].set_ylim(0, v_ceil)

for f in ric:
    if 0 < f < len(t_v):
        axs[1].axvline(t_v[f - 1], color=BLUE, linestyle=':', alpha=0.75, lw=1.2)
for f in lic:
    if 0 < f < len(t_v):
        axs[1].axvline(t_v[f - 1], color=RED,  linestyle=':', alpha=0.75, lw=1.2)

axs[2].plot(t_a, a_r, color=BLUE, label='Right Heel Accel', lw=1.8)
axs[2].plot(t_a, a_l, color=RED,  label='Left  Heel Accel', lw=1.8)
axs[2].axhline(0, color='gray', lw=0.8)   
axs[2].set_ylabel('Acceleration (px/s²)')
axs[2].set_xlabel('Time (s)')
axs[2].set_title('Heel Acceleration Magnitude')
axs[2].legend(loc='upper right')
axs[2].grid(alpha=0.35)

a_ceil  = max(np.percentile(a_r_mid, 99), np.percentile(a_l_mid, 99)) * 1.3
a_floor = min(np.percentile(a_r_mid,  1), np.percentile(a_l_mid,  1)) * 1.3
axs[2].set_ylim(a_floor, a_ceil)

for f in ric:
    if 0 < f - 1 < len(t_a):
        axs[2].axvline(t_a[f - 2], color=BLUE, linestyle=':', alpha=0.75, lw=1.2)
for f in lic:
    if 0 < f - 1 < len(t_a):
        axs[2].axvline(t_a[f - 2], color=RED,  linestyle=':', alpha=0.75, lw=1.2)

plt.tight_layout()

output_plot = r'C:\Users\Arnav Ranjan\.gemini\antigravity-ide\brain\c7e04f53-a826-4e08-8546-08e56cb30662\kinematics_plot.png'
plt.savefig(output_plot, dpi=150)
plt.show()
print(f"Plot saved → {output_plot}")