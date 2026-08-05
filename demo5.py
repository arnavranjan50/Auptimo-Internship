from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import cv2
import mediapipe as mp
import numpy as np
import wx
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
mp_pose = mp.solutions.pose

def apply_savgol_filter(signal, fps):

    signal = np.asarray(signal, dtype=float)

    if len(signal) < 7:
        return signal

    # Same window logic used in your existing function
    window = max(27, int(round(0.25 * fps)))

    if window % 2 == 0:
        window += 1

    # Window cannot be larger than the signal
    if window > len(signal):
        window = (
            len(signal)
            if len(signal) % 2 == 1
            else len(signal) - 1
        )

    if window < 3:
        return signal

    return savgol_filter(signal, window, 2)

def find_stable_frames(x_values, frame_numbers, frame_width, fps):
    print(x_values, frame_numbers, frame_width, fps, "bwahahah")
    x_values = np.array(x_values) * frame_width
    frame_numbers = np.array(frame_numbers)

    win = max(2, int(round(fps / 6)))
    #slope_threshold = 0.2083
    slope_threshold = 0.349066
    required_ratio = 0.40

    stable_frames = set()

    if len(x_values) < win:
        return []

    # Start from the first detected frame and move until the end
    for start in range(len(x_values) - win + 1):

        end = start + win
        slopes = []

        # Find slopes inside the current window
        for j in range(start + 1, end):

            frame_difference = frame_numbers[j] - frame_numbers[j - 1]

            if frame_difference > 0:

                slope = abs((x_values[j] - x_values[j - 1]) / frame_difference)
                slopes.append((slope, frame_numbers[j - 1], frame_numbers[j]))

        if len(slopes) == 0:
            continue

        good_slopes = 0

        for slope, previous_frame, current_frame in slopes:

            if slope < slope_threshold:
                good_slopes += 1

        ratio = good_slopes / len(slopes)

        # The window is stable when at least 60% slopes
        # are below the threshold
        if ratio >= required_ratio:

            for slope, previous_frame, current_frame in slopes:

                if slope < slope_threshold:
                    stable_frames.add(int(previous_frame))
                    stable_frames.add(int(current_frame))

    return sorted(stable_frames)


class VideoApp(wx.Frame):

    def __init__(self):

        super().__init__(
            None,
            title="Ankle Stability Frame Finder",
            size=(700, 500)
        )

        self.video_path = ""

        panel = wx.Panel(self)

        load_button = wx.Button(
            panel,
            label="Load Video"
        )

        play_button = wx.Button(
            panel,
            label="Play Video"
        )

        tracking_button = wx.Button(
            panel,
            label="Start Tracking"
        )

        self.status = wx.StaticText(
            panel,
            label="No video selected"
        )

        self.output = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY
        )

        button_layout = wx.BoxSizer(
            wx.HORIZONTAL
        )

        button_layout.Add(
            load_button,
            1,
            wx.ALL,
            5
        )

        button_layout.Add(
            play_button,
            1,
            wx.ALL,
            5
        )

        button_layout.Add(
            tracking_button,
            1,
            wx.ALL,
            5
        )

        main_layout = wx.BoxSizer(
            wx.VERTICAL
        )

        main_layout.Add(
            button_layout,
            0,
            wx.EXPAND | wx.ALL,
            10
        )

        main_layout.Add(
            self.status,
            0,
            wx.LEFT | wx.RIGHT | wx.BOTTOM,
            15
        )

        main_layout.Add(
            self.output,
            1,
            wx.EXPAND | wx.ALL,
            10
        )

        panel.SetSizer(main_layout)

        load_button.Bind(
            wx.EVT_BUTTON,
            self.load_video
        )

        play_button.Bind(
            wx.EVT_BUTTON,
            self.play_video
        )

        tracking_button.Bind(
            wx.EVT_BUTTON,
            self.start_tracking
        )

        self.Centre()
        self.Show()


    def load_video(self, event):

        dialog = wx.FileDialog(
            self,
            "Select Video",
            wildcard=(
                "Video files "
                "(*.mp4;*.avi;*.mov;*.mkv)|"
                "*.mp4;*.avi;*.mov;*.mkv"
                ),
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        )

        if dialog.ShowModal() == wx.ID_OK:

            self.video_path = dialog.GetPath()

            self.status.SetLabel(
                self.video_path
            )

            self.output.SetValue("")

        dialog.Destroy()


    def play_video(self, event):

        if not self.video_path:

            wx.MessageBox(
                "Please load a video first."
            )

            return

        video = cv2.VideoCapture(
            self.video_path
        )

        fps = video.get(
            cv2.CAP_PROP_FPS
        )

        if fps > 0:
            delay = int(1000 / fps)
        else:
            delay = 30

        while True:

            success, frame = video.read()

            if not success:
                break

            cv2.imshow(
                "Video - Press Q to stop",
                frame
            )

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

        video.release()
        cv2.destroyAllWindows()


    def start_tracking(self, event):

        if not self.video_path:

            wx.MessageBox(
                "Please load a video first."
            )

            return

        video = cv2.VideoCapture(
            self.video_path
        )

        fps = video.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30

        frame_width = int(
            video.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        frame_number = 0

        left_ankle_x = []
        left_frame_numbers = []

        right_ankle_x = []
        right_frame_numbers = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
            ) as pose:

            while True:

                success, frame = video.read()

                if not success:
                    break

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                result = pose.process(
                    rgb_frame
                )

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks.landmark

                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    left_ankle_x.append(left_ankle.x)
                    left_frame_numbers.append(frame_number)
                    right_ankle_x.append(right_ankle.x)
                    right_frame_numbers.append(frame_number)

                    height, width = frame.shape[:2]

                    #cv2.circle(
                        #frame,
                        #(
                            #int(left_ankle.x * width),
                            #int(left_ankle.y * height)
                            #),
                        #6,
                        #(0, 255, 0),
                        #-1
                    #)

                    #cv2.circle(
                        #frame,
                        #(
                            #int(right_ankle.x * width),
                            #int(right_ankle.y * height)
                            #),
                        #6,
                        #(0, 0, 255),
                        #-1
                    #)

                cv2.putText(
                    frame,
                    f"Frame: {frame_number}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Ankle Tracking - Press Q to stop",
                    frame
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_number += 1

        video.release()
        cv2.destroyAllWindows()
                
        # Filter the complete ankle X signals
        left_ankle_x_filtered = apply_savgol_filter(left_ankle_x, fps)
        right_ankle_x_filtered = apply_savgol_filter(right_ankle_x, fps)        

        left_stable_frames = find_stable_frames(left_ankle_x_filtered, left_frame_numbers, frame_width, fps)        
        right_stable_frames = find_stable_frames(right_ankle_x_filtered, right_frame_numbers, frame_width, fps)
        
        print("Left ankle stable frames:", left_stable_frames)
        print("Right ankle stable frames:", right_stable_frames)        
        # Plot ankle X against frame number
        plt.figure(figsize=(10, 6))
        poly = PolynomialFeatures(degree=4)

        calc_norm = np.array([
            0.58, 
            0.4, 
            0.52, 
            0.54, 
            0.56, 
            0.54, 
            0.383333333, 
            0.366666667, 
            0.366666667, 
            0.35, 
            0.366666667, 
            0.366666667, 
            0.35, 
            0.466666667, 
            0.483333333, 
            0.483333333, 
            0.5, 
            0.433333333, 
            0.383333333, 
            0.316666667,    
            0.38, 
            0.34, 
            0.34, 
            0.32, 
            0.32, 
            0.383333333, 
            0.4, 
            0.366666667, 
            0.44, 
            0.36, 
            0.34, 
            0.4
        ])

        pred_norm = np.array([
            0.46, 
            0.26, 
            0.42, 
            0.46, 
            0.44, 
            0.44,    
            0.3, 
            0.283333333, 
            0.266666667, 
            0.266666667,    
            0.3, 
            0.283333333, 
            0.183333333, 
            0.383333333, 
            0.383333333, 
            0.366666667, 
            0.383333333, 
            0.333333333, 
            0.283333333, 
            0.216666667,    
            0.04, 
            0.24, 
            0.26, 
            0.22, 
            0.2, 
            0.283333333, 
            0.3, 
            0.3, 
            0.34, 
            0.24, 
            0.24, 
            0.3, 
        ])
        X = pred_norm.reshape(-1, 1)
        y = calc_norm

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_quad = model.predict(X_poly)

        sort_idx = np.argsort(pred_norm)

        plt.scatter(pred_norm, calc_norm,
            color='blue',
            label='Data')

        plt.plot(pred_norm[sort_idx],
         y_quad[sort_idx],
         'r-',
         linewidth=2.5,
         label='Quadratic Regression')

       
        
        plt.title("Left and Right Ankle X Tracking")
        plt.xlabel("Frame Number")
        plt.ylabel("Ankle X Value")
        
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()        

        


app = wx.App()

VideoApp()

app.MainLoop()