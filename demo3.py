from mediapipe.python import solution_base
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import numpy as np
import os

# -----------------------------
# Butterworth Low-pass Filter
# -----------------------------
def butter_lowpass_filter(signal, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

video_path = r"C:\Users\Arnav Ranjan\OneDrive\Desktop\GaitON Internship\lat (online-video-cutter.com).mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("Could not open video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")

# -----------------------------
# Storage
# -----------------------------
frame_numbers = []
left_knee_x = []
left_knee_y = []
right_knee_x = []
right_knee_y = []

frame_idx = 0

# -----------------------------
# Pose Estimator
# -----------------------------
if (os.path.exists("left_knee_x.npy") and
    os.path.exists("left_knee_y.npy") and
    os.path.exists("right_knee_x.npy") and
    os.path.exists("right_knee_y.npy")):

    print("Loading coordinates...")

    left_knee_x = np.load("left_knee_x.npy")
    left_knee_y = np.load("left_knee_y.npy")
    right_knee_x = np.load("right_knee_x.npy")
    right_knee_y = np.load("right_knee_y.npy")

    frame_numbers = np.arange(len(left_knee_x))

else:
    with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:

     while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            # Pixel coordinates
            left_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * width
            left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * height
            right_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width
            right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height

            left_knee_x.append(left_x)
            left_knee_y.append(left_y)

            right_knee_x.append(right_x)
            right_knee_y.append(right_y)
            frame_numbers.append(frame_idx)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2,
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 0, 255),
                    thickness=2,
                ),
            )
        else:
            print("No detection")

        cv2.imshow("Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()

np.save("left_knee_x.npy", np.array(left_knee_x))
np.save("left_knee_y.npy", np.array(left_knee_y))
np.save("right_knee_x.npy", np.array(right_knee_x))
np.save("right_knee_y.npy", np.array(right_knee_y))

print("Coordinates saved...")

left_knee_x = np.load("left_knee_x.npy")
left_knee_y = np.load("left_knee_y.npy")
right_knee_x = np.load("right_knee_x.npy")
right_knee_y = np.load("right_knee_y.npy")

frame_numbers = np.arange(len(left_knee_x))
# -----------------------------
# Filtering
# -----------------------------
# Window length must be odd and less than number of samples

cutoff = 5     
order = 2

left_x_filtered = butter_lowpass_filter(
    np.array(left_knee_x), cutoff, fps, order)

left_y_filtered = butter_lowpass_filter(
    np.array(left_knee_y), cutoff, fps, order)

right_x_filtered = butter_lowpass_filter(
    np.array(right_knee_x), cutoff, fps, order)

right_y_filtered = butter_lowpass_filter(
    np.array(right_knee_y), cutoff, fps, order)
# -----------------------------
# Plot
# -----------------------------
fig, axs = plt.subplots(2, 1, figsize=(11, 13), sharex=True)

# ===========================
# Position
# ===========================
axs[0].plot(frame_numbers, left_knee_x,
         color='lightcoral',
         linewidth=2,
         alpha = 0.4,
         label='Left Knee (Raw)')

axs[0].plot(frame_numbers, left_x_filtered,
         color='red',
         linewidth=2,
         label='Left Knee')

axs[0].plot(frame_numbers, right_knee_x,
         color='skyblue',
         linewidth=2,
         alpha=0.4,
         label='Right Knee (Raw)')

axs[0].plot(frame_numbers, right_x_filtered,
         color='blue',
         linewidth=2,
         label='Right Knee')

axs[0].set_ylabel("Position (pixels)")
axs[0].set_title("Knee Position")
axs[0].grid(True)
axs[0].legend(loc='upper right')

# Optional because image coordinates increase downward
#axs[0].invert_yaxis()



# -----------------------------
# Velocity
# -----------------------------
dt = 1 / fps

left_vx = np.gradient(left_x_filtered, dt)
left_vy = np.gradient(left_y_filtered, dt)

right_vx = np.gradient(right_x_filtered, dt)
right_vy = np.gradient(right_y_filtered, dt)

# -----------------------------
# Velocity Angle
# -----------------------------
left_angle = np.abs(np.degrees(np.arctan2(1,left_vx)))
right_angle = np.abs(np.degrees(np.arctan2(1,right_vx)))

left_velocity = left_vx
right_velocity = right_vx

print("-" * 45)

for i in range(len(frame_numbers)):
    print(f"{frame_numbers[i]:4d}\t{left_angle[i]:7.2f}\t\t{right_angle[i]:7.2f}")

left_angle_array = []
i = 0

while i < len(left_angle):
    array = []

    while i < len(left_angle) and left_angle[i] <= 10:
        array.append(i)
        i += 1

    if len(array) >= 10:
        left_angle_array.append(array)

    if len(array) == 0:
        i += 1

print("Left:", left_angle_array)

right_angle_array = []
i = 0

while i < len(right_angle):
    array = []

    while i < len(right_angle) and right_angle[i] <= 10:
        array.append(i)
        i += 1

    if len(array) >= 10:
        right_angle_array.append(array)

    if len(array) == 0:
        i += 1

print("Right:", right_angle_array)

from scipy.stats import linregress

window = 5

angles = []

for i in range(len(left_vx) - window):
    x = frame_numbers[i:i+window]
    y = left_vx[i:i+window]

    slope = linregress(x, y).slope
    angle = np.degrees(np.arctan(slope))
    angles.append(angle)

print(slope, angles)


axs[1].plot(frame_numbers,
         left_velocity,
         color='red',
         linewidth=2,
         label='Left Ankle Velocity')

axs[1].plot(frame_numbers,
         right_velocity,
         color='blue',
         linewidth=2,
         label='Right Ankle Velocity')

axs[1].set_xlabel("Frame")
axs[1].set_ylabel("Velocity (pixels/s)")
axs[1].set_title("Ankle Velocity")
axs[1].grid(True)
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.show()

#import time
#import numpy as np
#from rtmlib import Body, Custom, PoseTracker, draw_skeleton
#import cv2
#from scipy.signal import butter, filtfilt
#import matplotlib.pyplot as plt


#cap = cv2.VideoCapture(r"C:\Users\Auptimo\Downloads\lat (online-video-cutter.com).mp4")  # for video file instead of webcam, use cap = cv2.VideoCapture('./demo.mp4')

#device = 'cpu'
#backend = 'onnxruntime'
#openpose_skeleton = False

##pose_tracker = PoseTracker(Body,
                        ##mode='balanced',
                        ##det_frequency=10,  # detect every 10 frames
                        ##backend=backend, device=device,
                        ##to_openpose=False,
                        ##tracking=False)

## # Or with a custom class
#from functools import partial
#custom = partial(Custom,
                #det_class='YOLOX',
                #det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
                #det_input_size=(640, 640),
                #pose_class='RTMPose',
                #pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
                #pose_input_size=(192, 256))
#pose_tracker = PoseTracker(custom,
                        #det_frequency=10,
                        #backend=backend, device=device,
                        #to_openpose=False,
                        #tracking = False)

#frame_idx = 0
#fps = cap.get(cv2.CAP_PROP_FPS)
#dt = 1.0 / cap.get(cv2.CAP_PROP_FPS)
#r_heel, l_heel, r_toe, l_toe, timestamps = [], [], [], [], []
#prev_time = time.time()

#def calculate_velocity_graph_angle(velocity_data):
    ##y_range = np.max(velocity_data) - np.min(velocity_data)
    ##x_range = len(velocity_data)
    ##if y_range == 0: y_range = 1 
    
    ##dy = np.diff(velocity_data)
    ##dy_percent = dy / y_range
    ##dx_percent = 1.0 / x_range
    ##visual_ratio = 4.0 / 11.0
    
    ##return np.degrees(np.arctan2(dy_percent * visual_ratio, dx_percent))
    
    #angles_rad = np.arctan(np.diff(velocity_data))
    #angles_deg = np.degrees(angles_rad)
    
    #return angles_deg

#def smooth_1d(data, window_size=5):
    #if len(data) < window_size: return data
    #smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    #pad = window_size // 2
    #smoothed[:pad] = data[:pad]
    #smoothed[-pad:] = data[-pad:]
    #return smoothed

#def clean_coordinates(coords):
    #"""
    #Prevents 'Teleportation Spikes'. 
    #If the AI loses tracking and outputs [0,0], we freeze the point at its last known location.
    #"""
    #cleaned = np.copy(coords)
    ## Forward fill missing data
    #for i in range(1, len(cleaned)):
        #if cleaned[i, 0] == 0 and cleaned[i, 1] == 0:
            #cleaned[i] = cleaned[i-1]
    ## Backward fill the start if needed
    #for i in range(len(cleaned)-2, -1, -1):
        #if cleaned[i, 0] == 0 and cleaned[i, 1] == 0:
            #cleaned[i] = cleaned[i+1]
    #return cleaned

#def lowpass_filter_coords(coords, cutoff, fs, order=2):
    #if len(coords) < 15: 
        #return coords
    #b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
    #return np.column_stack((filtfilt(b, a, coords[:, 0]), filtfilt(b, a, coords[:, 1])))


#while cap.isOpened():
    #success, frame = cap.read()
    #frame_idx += 1
    #if not success:
        #break

    #keypoints, scores = pose_tracker(frame)
    
    #timestamps.append(frame_idx)

    #if len(keypoints) > 0:
        #kpts = keypoints[0]
        #r_heel.append(kpts[25]); l_heel.append(kpts[24])
        #r_toe.append(kpts[21]);  l_toe.append(kpts[20])
    #else:
        #for arr in (r_heel, l_heel, r_toe, l_toe):
            #arr.append(arr[-1] if arr else np.array([0.0, 0.0]))

    #canvas = draw_skeleton(frame, keypoints, scores, kpt_thr=0.3)

    #curr_time = time.time()
    #process_fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    #prev_time = curr_time
    #print(f"\rProcessing Video... FPS: {process_fps:.1f}   ", end="", flush=True)

    #cv2.imshow('Pose Estimation', canvas)
    #if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'): 
        #break
    
#print(f"\n--- Processing Full Segment (Total: {len(timestamps)} frames) ---")

## Clean the off-screen [0,0] coordinates before applying any physics!
#r_heel = clean_coordinates(r_heel)
#l_heel = clean_coordinates(l_heel)
#r_toe = clean_coordinates(r_toe)
#l_toe = clean_coordinates(l_toe)
#print(len(r_heel))
#sr_heel = lowpass_filter_coords(r_heel, 4, fps)
#sl_heel = lowpass_filter_coords(l_heel, 4, fps)
##sr_toe = lowpass_filter_coords(r_toe, 6, fps)
##sl_toe = lowpass_filter_coords(l_toe, 6, fps)

#vr_heel = np.sqrt(np.diff(sr_heel[:, 0])**2 + np.diff(sr_heel[:, 1])**2) / dt
#vl_heel = np.sqrt(np.diff(sl_heel[:, 0])**2 + np.diff(sl_heel[:, 1])**2) / dt
#ar_heel = np.diff(vr_heel) / dt
#al_heel = np.diff(vl_heel) / dt

#print(f"\n--- Velocity Array Lengths: Right={len(vr_heel)}, Left={len(vl_heel)} ---")
#print(vr_heel)
#angle_vr_heel = calculate_velocity_graph_angle(vr_heel)
#angle_vl_heel = calculate_velocity_graph_angle(vl_heel)

#print(f"\n--- Heel Velocity Graph Angles (Full Data: {len(angle_vr_heel)} Segments) ---")
#for i in range(min(len(angle_vr_heel), len(angle_vl_heel))):
    #print(f"Frames ({i}-{i+1}) | Right: {angle_vr_heel[i]:7.2f}° | Left: {angle_vl_heel[i]:7.2f}°")

##ric = find_heel_strikes(sr_heel[:, 1])
##lic = find_heel_strikes(sl_heel[:, 1])

#fig, axs = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
#print(timestamps)
#axs[0].plot(timestamps, sr_heel[:, 1], label='Right Heel', color='#1f77b4', linewidth=2)
#axs[0].plot(timestamps, sl_heel[:, 1], label='Left Heel', color='#d62728', linewidth=2)
##axs[0].invert_yaxis()
#axs[0].set_title('Vertical Joint Trajectories')
#axs[0].set_ylabel('Position (Pixels)')

#axs[1].plot(timestamps[1:], vr_heel, label='Right Vel', color='#1f77b4', linewidth=2)
#axs[1].plot(timestamps[1:], vl_heel, label='Left Vel', color='#d62728', linewidth=2)
#axs[1].axvline(x=40, color='red', linestyle='--', linewidth=2)
#axs[1].set_title('Heel Velocity Magnitude')
#axs[1].set_ylabel('Velocity (px/s)')

#axs[2].plot(timestamps[2:], ar_heel, label='Right Accel', color='#1f77b4', linewidth=2)
#axs[2].plot(timestamps[2:], al_heel, label='Left Accel', color='#d62728', linewidth=2)
#axs[2].set_title('Heel Acceleration Magnitude')
#axs[2].set_ylabel('Acceleration (px/s²)')
#axs[2].set_xlabel('Time (seconds)')


#for ax in axs: 
    #ax.grid(True, alpha=0.5)
    #ax.legend(loc='upper right')

#plt.tight_layout()
#plt.show() 