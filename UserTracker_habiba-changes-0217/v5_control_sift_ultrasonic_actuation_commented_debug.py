import cv2
import mediapipe as mp
import serial
import sys
import time
import numpy as np
import threading
import os
import subprocess
import socket
from datetime import datetime
import lgpio

class VisualFeatureTracker:
    """
    This class tracks an object (the calibrated user) using SIFT feature detection.
    After calibration, we store descriptors from the bounding box region of the user.
    At runtime, we extract SIFT features from a downsampled frame at a specified polling rate
    and try to match them with the stored descriptors. If enough matches are found (based on
    a threshold), we consider it the same user and compute a new bounding box.
    """
    def __init__(
        self,
        min_keypoints_threshold,
        ratio_threshold,
        max_features=500,
        downsample_scale=1.0,
        sift_interval=0
    ):
        """
        Initialize the SIFT detector, FLANN matcher, and various tracking states.

        :param min_keypoints_threshold: Number of matching keypoints needed to confirm it's the same user.
        :param ratio_threshold: Lowe's ratio test threshold for confirming a match.
        :param max_features: Limit on the number of SIFT keypoints.
        :param downsample_scale: Scale factor to resize frames before computing features (0 < scale <= 1).
        :param sift_interval: Time interval (seconds) between consecutive SIFT computations.
        """
        # Create SIFT with a capped number of features.
        self.sift = cv2.SIFT_create(nfeatures=max_features)

        # Use a KD-tree based search for FLANN.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Descriptors for the user's ROI at calibration.
        self.object_descriptors = None
        self.prev_bbox = None
        self.initialized = False

        # Thresholds and parameters.
        self.min_keypoints_threshold = min_keypoints_threshold
        self.ratio_threshold = ratio_threshold
        self.downsample_scale = downsample_scale
        self.sift_interval = sift_interval

        # Track when we last ran SIFT.
        self.last_sift_time = 0

    def initialize_object_tracking(self, frame, bbox):
        """
        Extract SIFT features from the bounding box region at calibration time.
        Store them for future matching.

        :param frame: The image (from the camera) in which the user stands.
        :param bbox: The bounding box (x, y, w, h) representing the user.
        """
        self.prev_bbox = bbox
        x, y, w, h = bbox
        user_roi = frame[y:y+h, x:x+w]
        keypoints, descriptors = self.sift.detectAndCompute(user_roi, None)

        if descriptors is None or len(descriptors) == 0:
            print("[Debug] No descriptors found during calibration ROI.")
            return

        # Store descriptors for subsequent re-identification.
        self.object_descriptors = descriptors
        self.initialized = True
        print(f"[Debug] SIFT calibration: Stored {len(descriptors)} descriptors.")

    def track_object_in_frame(self, frame):
        """
        Attempt to match the stored descriptors with the SIFT features extracted from
        a downsampled version of the frame, but only at the specified polling rate.
        If enough matches are found, compute a bounding box around those matched keypoints.

        :param frame: The full image in which to detect the user.
        :return: The bounding box of the matched keypoints if recognized; otherwise None.
        """
        if not self.initialized or self.object_descriptors is None:
            print("[Debug] track_object_in_frame() called, but tracker not initialized.")
            return None

        # Check if enough time has passed to run SIFT.
        now = time.time()
        if (now - self.last_sift_time) < self.sift_interval:
            # Skip SIFT this time.
            return None

        self.last_sift_time = now

        # Downsample frame.
        if self.downsample_scale != 1.0:
            downsampled_frame = cv2.resize(
                frame,
                None,
                fx=self.downsample_scale,
                fy=self.downsample_scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            downsampled_frame = frame

        # Compute SIFT features on the downsampled frame.
        keypoints, descriptors = self.sift.detectAndCompute(downsampled_frame, None)
        if descriptors is None or len(descriptors) == 0:
            print("[Debug] No descriptors found in the current downsampled frame.")
            return None

        try:
            matches = self.flann.knnMatch(descriptors, self.object_descriptors, k=2)
        except Exception as e:
            # If matching fails for any reason, mark as not tracked.
            print(f"[Debug] FLANN matching failed: {e}")
            return None

        good_points = []
        for i, pair in enumerate(matches):
            if len(pair) == 2:
                d1, d2 = pair[0].distance, pair[1].distance
                # Avoid division by zero.
                if d2 == 0:
                    continue
                ratio = d1 / d2
                if ratio < self.ratio_threshold:
                    # If pass ratio test, record the location of this keypoint.
                    kp_x, kp_y = keypoints[i].pt

                    # Scale back up to original frame size.
                    if self.downsample_scale != 1.0:
                        kp_x /= self.downsample_scale
                        kp_y /= self.downsample_scale

                    good_points.append((kp_x, kp_y))

        print(f"[Debug] SIFT re-ID: Found {len(good_points)} good matches.")
        # Check if we have enough matches to conclude it's the same user.
        if len(good_points) < self.min_keypoints_threshold:
            print("[Debug] Not enough matches to confirm re-identification.")
            return None

        # Compute bounding box around matched keypoints.
        pts = np.array(good_points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        new_bbox = (
            int(x_min),
            int(y_min),
            int(x_max - x_min),
            int(y_max - y_min)
        )
        self.prev_bbox = new_bbox
        print(f"[Debug] SIFT re-ID successful. Updated bounding box: {new_bbox}")
        return new_bbox

class CameraManager:
    """
    Handles camera setup, frame capture, and cleanup. Uses a dedicated thread to read frames
    from the camera continuously and makes the most recent frame available for retrieval.
    """
    def __init__(self, camera_index=0, width=480, height=270, max_retries=5, delay=1):
        # Basic camera parameters.
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.max_retries = max_retries
        self.delay = delay

        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.running = False

        # Make sure no other process is using this camera device.
        self.release_camera_resource_if_busy()
        self.init_camera()

    def release_camera_resource_if_busy(self):
        """
        Checks if /dev/video0 is busy, and if so, forcibly kills the process.
        This ensures the camera resource is free before we proceed.
        """
        cam_process = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
        if cam_process.stdout.strip():
            os.system("sudo fuser -k /dev/video0")
            time.sleep(1)

    def init_camera(self):
        """
        Attempt to open the camera (CAP_V4L2). If it fails, retry up to max_retries times,
        each time releasing any hold on /dev/video0.
        """
        for attempt in range(1, self.max_retries + 1):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print(f"[Debug] Camera opened successfully on attempt {attempt}.")
                return
            self.release_camera_resource_if_busy()
            time.sleep(self.delay)
        self.cap = None
        print("[Error] Failed to open camera after multiple retries.")

    def start_capture(self):
        """
        Spawns a separate daemon thread to continuously capture frames from the camera.
        """
        if self.cap is None:
            print("[Error] Cannot start capture because camera is not opened.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames_loop, daemon=True)
        self.thread.start()
        print("[Debug] Started camera capture thread.")

    def _capture_frames_loop(self):
        """
        The threaded loop that runs in the background, reading frames and storing
        the latest valid frame.
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def get_latest_frame(self):
        """
        Thread-safe retrieval of the most recently read camera frame.
        :return: The latest captured frame or None if not available.
        """
        with self.lock:
            return self.latest_frame

    def stop_capture(self):
        """
        Signals the thread to stop running and releases the camera.
        """
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        print("[Debug] Camera capture stopped and camera released.")


class PoseEstimator:
    """
    Wraps Mediapipe's Pose solution for easy pose detection.
    """
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        print("[Debug] PoseEstimator initialized.")

    def estimate_pose(self, frame):
        """
        Given a BGR frame, converts it to RGB and runs Mediapipe Pose detection.

        :param frame: BGR image to process.
        :return: The results object containing pose landmarks.
        """
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)


class UltrasonicManager:
    """
    Manages multiple ultrasonic sensors and measures distances from them.
    Uses a multi-threaded approach to measure each sensor concurrently.
    """

    # Hardcoded sensor pins: front and left.
    SENSORS = {
        "front": {"TRIG": 11, "ECHO": 12},
        "left":  {"TRIG": 13, "ECHO": 16},
    }

    TIMEOUT = 0.04  # Timeout in seconds to avoid an infinite loop if no echo is received.

    def __init__(self):
        # For each sensor, record TRIG/ECHO.
        self.sensors = {name: {"TRIG": cfg["TRIG"], "ECHO": cfg["ECHO"]} for name, cfg in UltrasonicManager.SENSORS.items()}
        self.cached_distances = {name: None for name in self.sensors}
        self.lock = threading.Lock()
        self.chip = lgpio.gpiochip_open(0)
        # Initialize each sensor's pins.
        for name, pins in self.sensors.items():
            lgpio.gpio_claim_output(self.chip, pins["TRIG"])
            lgpio.gpio_claim_input(self.chip, pins["ECHO"])
            lgpio.gpio_write(self.chip, pins["TRIG"], 0)
        print("[Debug] UltrasonicManager initialized.")

    def measure_single_distance(self, trig, echo):
        """
        Triggers the ultrasonic sensor and measures the pulse duration on the echo pin.
        Then calculates the distance based on the speed of sound.
        """
        lgpio.gpio_write(self.chip, trig, 0)
        time.sleep(0.000002)
        lgpio.gpio_write(self.chip, trig, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(self.chip, trig, 0)

        start_time = time.time()
        # Wait for the echo pin to go HIGH (pulse start).
        while lgpio.gpio_read(self.chip, echo) == 0:
            if time.time() - start_time > UltrasonicManager.TIMEOUT:
                return None
        pulse_start = time.time()
        # Wait for the echo pin to go LOW (pulse end).
        while lgpio.gpio_read(self.chip, echo) == 1:
            if time.time() - pulse_start > UltrasonicManager.TIMEOUT:
                return None
        pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        # speed of sound (34300 cm/s) => distance = (time * speed)/2
        distance = (pulse_duration * 34300) / 2
        return round(distance, 2)

    def _distance_measure_thread(self, sensor_name, trig, echo):
        """
        Worker thread method for measuring distance on a single ultrasonic sensor.
        The distance is stored in a shared dictionary.
        """
        dist = self.measure_single_distance(trig, echo)
        with self.lock:
            self.cached_distances[sensor_name] = dist

    def get_all_distances(self):
        """
        Launches a thread for each sensor to perform distance measurements.
        Waits for all threads to finish and returns the cached measurements.
        """
        threads = []
        for name, pins in self.sensors.items():
            t = threading.Thread(target=self._distance_measure_thread, args=(name, pins["TRIG"], pins["ECHO"]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return self.cached_distances

    def cleanup_gpio(self):
        """
        Closes the gpio chip resource.
        """
        lgpio.gpiochip_close(self.chip)
        print("[Debug] UltrasonicManager GPIO cleaned up.")


class DistanceEstimator:
    """
    Calibrates the distance to a user based on a reference distance and the observed
    vertical height of the user (e.g., shoulders to hips). Then estimates distance
    at runtime if the user's vertical height changes.
    """
    def __init__(self, reference_distance):
        self.reference_distance = reference_distance
        self.reference_height = None
        print(f"[Debug] DistanceEstimator initialized with reference_distance={reference_distance}.")

    def calibrate_height(self, vertical_height):
        """
        Store a baseline height from shoulders to hips when the user is at the reference distance.

        :param vertical_height: The vertical distance in pixels between shoulders and hips.
        """
        self.reference_height = vertical_height
        print(f"[Debug] Calibrated reference height: {vertical_height} pixels.")

    def estimate_distance_from_height(self, current_height):
        """
        Estimate the current distance based on how the user's vertical height in the frame
        compares to the baseline calibrated height.

        :param current_height: The user's current vertical height in pixels.
        :return: The estimated distance in meters (assuming reference_distance is in meters).
        """
        if self.reference_height is not None and current_height > 0:
            est_dist = round(self.reference_distance * (self.reference_height / current_height), 2)
            return est_dist
        return None

class MotionController:
    """
    Computes left and right wheel velocities based on distance and angle error.
    Adds a priority for rotation, meaning if angle is large, forward speed is reduced.
    Also includes smoothing for both angular and linear velocities. 
    Finally, it transforms user center offsets to an approximate angle.
    """
    def __init__(
        self,
        kv,
        kw,
        target_distance,
        distance_tolerance=0.1,
        angle_tolerance_deg=25,
        rotation_priority=True,
        smoothing_alpha=0.2
    ):
        """
        :param rotation_priority: If True, the linear velocity is scaled down when angle offset is large.
        :param smoothing_alpha: Low-pass filter factor for velocity smoothing (both linear and angular).
        """
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        # Gains for linear (kv) and angular (kw) velocity.
        self.kv = kv
        self.kw = kw
        self.rotation_priority = rotation_priority
        self.smoothing_alpha = smoothing_alpha

        # Keep track of previous velocities for smoothing.
        self.previous_linear_vel = 0.0
        self.previous_angular_vel = 0.0

        # This factor helps convert between linear velocity and wheel speed.
        self.wheelbase_factor = 0.5

    def compute_wheel_velocities(self, distance_est, user_center_x, frame_width, max_speed):
        """
        Given the user's estimated distance and horizontal position, compute actual wheel velocities.
        1) Convert center offset into an approximate angle.
        2) Compute raw linear_vel (with deadband) and raw angular_vel.
        3) Smooth both velocities to avoid sudden jumps.
        4) Apply rotation priority if requested.
        5) Convert to differential drive speeds (wl, wr) and clamp.
        """
        if frame_width == 0:
            return 0.0, 0.0, 0

        # 1) Convert horizontal offset to an approximate angle.
        offset = user_center_x - (frame_width / 2)
        norm_offset = offset / (frame_width / 2)
        angle_deg = max(min(norm_offset * 90, 90), -90)
        angle_int = int(angle_deg)

        # 2) Compute the raw linear velocity with a simple deadband.
        distance_err = 0 if distance_est is None else distance_est - self.target_distance

        if abs(distance_err) < self.distance_tolerance:
            raw_linear_vel = 0.0
        else:
            raw_linear_vel = self.kv * distance_err

        # 2a) Compute the raw angular velocity (pre-smoothing).
        raw_angular_vel = self.kw * angle_int

        # 3) Smooth both linear and angular velocities:
        #    linear_vel_new = alpha * (raw) + (1 - alpha) * (previous)
        alpha = self.smoothing_alpha
        linear_vel = alpha * raw_linear_vel + (1 - alpha) * self.previous_linear_vel
        angular_vel = alpha * raw_angular_vel + (1 - alpha) * self.previous_angular_vel

        # Store for the next iteration
        self.previous_linear_vel = linear_vel
        self.previous_angular_vel = angular_vel

        # 4) Clamp angular velocity to max_speed.
        if abs(angular_vel) > max_speed:
            angular_vel = np.sign(angular_vel) * max_speed

        # 5) If rotation priority is enabled, reduce linear velocity if angle is large.
        if self.rotation_priority:
            threshold = float(self.angle_tolerance_deg)
            scale_factor = max(0.0, 1.0 - (abs(angle_int) / (threshold * 2)))
            linear_vel *= scale_factor

        # 6) Convert to left and right wheel speeds.
        base_speed = linear_vel / self.wheelbase_factor
        wl = angular_vel + base_speed
        wr = -angular_vel + base_speed

        # 7) Clamp wheel speeds to ±max_speed.
        wl = max(min(wl, max_speed), -max_speed)
        wr = max(min(wr, max_speed), -max_speed)

        return wl, wr, angle_int

class SerialCommandSender:
    """
    Placeholder class to send wheel velocities via a serial connection.
    Currently, only prints them for debugging if serial is not actually enabled.
    """
    def __init__(self, port='/dev/serial0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate)

    def send_wheel_velocities(self, wl, wr):
        """
        Format and send wheel velocities (wl, wr) to  ESP32 through serial.
        """
        msg = f"w_l:{-wr:.3f} w_r:{-wl:.3f}\n"
        self.ser.write(msg.encode('utf-8'))
        print("[Debug] Serial command:", msg)

    def close_connection(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class UserTrackerApp:
    """
    Primary class that unifies the camera, pose detection, SIFT-based re-identification, motion control,
    and ultrasonic sensor inputs.

    The high-level process:
        1) Calibrate by observing the user's pose and storing a reference distance.
        2) Optionally store SIFT descriptors from the bounding box around the user.
        3) In the main tracking loop:
           - Attempt pose detection with Mediapipe.
           - If pose is lost, attempt SIFT-based re-identification in the entire frame.
           - Compute wheel velocities based on distance and angle offset.
        4) Send velocities to a motor controller (through serial) or debug print.
    """
    def __init__(
        self,
        camera_index,
        target_distance,
        reference_distance,
        polling_interval,
        serial_port,
        baudrate,
        serial_enabled,
        ultrasonic_enabled,
        draw_enabled,
        debug_keys,
        kv,
        kw,
        max_speed,
        emergency_stop_distance=0.25
    ):
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()
        self.serial_enabled = serial_enabled
        self.ultrasonic_enabled = ultrasonic_enabled
        self.draw_enabled = draw_enabled
        self.debug_keys = debug_keys  # If True, capture keyboard input for debugging.
        self.pending_start = False
        self.pending_stop = False
        self.pending_calibrate = False

        # Tracking state: "TRACKING" or "LOST".
        self.tracking_state = "LOST"

        print("[Debug] Initializing CameraManager...")
        self.camera_manager = CameraManager(camera_index=camera_index)
        self.camera_manager.start_capture()
        print("[Debug] Initializing PoseEstimator...")
        self.pose_estimator = PoseEstimator()
        print("[Debug] Initializing DistanceEstimator...")
        self.distance_estimator = DistanceEstimator(reference_distance)
        print("[Debug] Initializing MotionController...")
        # Setup motion controller with proportional gains.
        # Now includes rotation priority and smoothing, plus converting offset->angle.
        self.motion_controller = MotionController(
            kv=kv,
            kw=kw,
            target_distance=target_distance,
            rotation_priority=True,
            smoothing_alpha=0.3
        )
        # Optionally connect to serial.
        if self.serial_enabled:
            self.serial_sender = SerialCommandSender(port=serial_port, baudrate=baudrate)
        else:
            self.serial_sender = None
        # Setup ultrasonic sensors.
        if self.ultrasonic_enabled:
            print("[Debug] Initializing UltrasonicManager...")
            self.ultrasonic = UltrasonicManager()
        else:
            self.ultrasonic = None
        # Setup SIFT-based re-identification tracker.
        print("[Debug] Initializing VisualFeatureTracker...")
        self.visual_tracker = VisualFeatureTracker(min_keypoints_threshold=5, ratio_threshold=0.7)

        self.window_name = "Distance & Angle Tracker"
        self.max_speed = max_speed  # used for clamping speeds.
        self.emergency_stop_distance = emergency_stop_distance  # obstacle distance (in meters)
        print("[Debug] UserTrackerApp initialization complete.")

    def extract_pose_metrics(self, results, frame):
        """
        Given Mediapipe results and a frame, compute:
          - The vertical height between shoulders and hips.
          - The center x-coordinate of the user.
          - Shoulder/hip landmarks.
          - The frame width and height.
        """
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        # Compute average Y for shoulders, average Y for hips.
        shoulder_y = (l_sh.y + r_sh.y) / 2 * h
        hip_y = (l_hip.y + r_hip.y) / 2 * h
        vert_height = abs(shoulder_y - hip_y)

        # Approximate center of the user horizontally.
        user_center_x = (l_sh.x + r_sh.x) / 2 * w
        return vert_height, user_center_x, l_sh, r_sh, l_hip, r_hip, w, h

    def command_socket_listener(self):
        """
        Sets up a UNIX domain socket to listen for external commands such as:
          - stop
          - calibrate
          - start:<distance>
        """
        COMMAND_SOCKET_PATH = "/tmp/command_socket"
        if os.path.exists(COMMAND_SOCKET_PATH):
            os.remove(COMMAND_SOCKET_PATH)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_sock:
            server_sock.bind(COMMAND_SOCKET_PATH)
            server_sock.listen(1)
            print("[Debug] Command socket listener started.")
            while True:
                conn, _ = server_sock.accept()
                with conn:
                    data = conn.recv(1024).decode().strip().lower()
                    if not data:
                        continue
                    parts = data.split(':')
                    cmd = parts[0]
                    if cmd == "stop":
                        self.request_stop()
                    elif cmd == "calibrate":
                        self.request_calibration()
                    elif cmd == "start":
                        distance_val = None
                        if len(parts) > 1:
                            try:
                                distance_val = float(parts[1])
                            except ValueError:
                                pass
                        self.request_start(distance_val)

    def request_stop(self):
        """
        Signals that we want to stop all tracking and motion.
        """
        print("[Debug] Stop requested.")
        if self.serial_enabled and self.serial_sender:
            self.serial_sender.send_wheel_velocities(0, 0)

        try:
            if self.pose_estimator and self.pose_estimator.pose:
                self.pose_estimator.pose.close()
                self.pose_estimator.pose = None
                print("[Debug] Closed MediaPipe Pose.")
        except Exception as e:
            print(f"[Warning] Error closing the MediaPipe Pose object: {e}")

        self.pending_stop = True

    def request_calibration(self):
        """
        Signals that a calibration step should occur.
        """
        print("[Debug] Calibration requested.")
        self.pending_calibrate = True

    def request_start(self, distance=None):
        """
        Signals that we want to start tracking.
        Optionally updates the target/reference distance.
        """

        print(f"[Debug] Start requested with distance={distance}.")
        if distance is not None:
            self.motion_controller.target_distance = distance
            self.distance_estimator.reference_distance = distance
        self.pending_start = True

    def send_status_update(self, status_message):
        """
        Send a status message to the /tmp/status_socket if it exists.
        """
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect("/tmp/status_socket")
                sock.sendall(status_message.encode())
        except FileNotFoundError:
            pass

    def calibrate_distance_reference(self):
        """
        During calibration:
         1) We wait for a valid pose detection.
         2) We measure the vertical height and store it as reference.
         3) We build a bounding box around shoulders and hips to feed into SIFT.
         4) Switch tracking state to "TRACKING" if successful.
        """
        print("[Debug] Starting calibration routine.")
        self.send_status_update("Calibration Started")
        while True:
            if self.pending_stop:
                self.send_status_update("Calibration Stopped")
                if self.serial_sender:
                    self.serial_sender.send_wheel_velocities(0, 0)
                print("[Debug] Calibration stopped externally.")
                break

            frame = self.camera_manager.get_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            if self.debug_keys:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.send_status_update("Calibration Stopped")
                    if self.serial_enabled and self.serial_sender:
                        self.serial_sender.send_wheel_velocities(0, 0)
                    print("[Debug] Calibration stopped via key.")
                    break

            results = self.pose_estimator.estimate_pose(frame)
            if results and results.pose_landmarks:
                # Extract user vertical height.
                vert_height, _, l_sh, r_sh, l_hip, r_hip, w, h = self.extract_pose_metrics(results, frame)
                self.distance_estimator.calibrate_height(vert_height)

                # Build bounding box from shoulder/hip landmarks.
                all_x = [l_sh.x * w, r_sh.x * w, l_hip.x * w, r_hip.x * w]
                all_y = [l_sh.y * h, r_sh.y * h, l_hip.y * h, r_hip.y * h]
                x_min, x_max = int(min(all_x)), int(max(all_x))
                y_min, y_max = int(min(all_y)), int(max(all_y))
                bb_w = x_max - x_min
                bb_h = y_max - y_min

                # Store SIFT descriptors from this bounding box.
                self.visual_tracker.initialize_object_tracking(frame, (x_min, y_min, bb_w, bb_h))
                print("[Info] SIFT calibration in progress.")

                # If the user wanted calibration, start the tracking loop next.
                if self.pending_calibrate:
                    # Once done, set state to TRACKING.
                    self.tracking_state = "TRACKING"
                    self.send_status_update("SIFT calibration complete: user descriptors stored.")
                    print("[Debug] Entering run_tracking_loop post-calibration.")
                    self.run_tracking_loop()
                    break

            if self.draw_enabled:
                debug_frame = frame.copy()
                cv2.putText(debug_frame, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.imshow(self.window_name, debug_frame)
                cv2.waitKey(1)

    def _attempt_pose(self, frame):
        """
        Attempts to detect user pose. If successful, returns wheel velocities,
        distance estimate, angle, etc. If unsuccessful, returns zero velocities.
        """
        results = self.pose_estimator.estimate_pose(frame)
        if results and results.pose_landmarks:
            vert_height, user_center_x, l_sh, r_sh, l_hip, r_hip, w, h = self.extract_pose_metrics(results, frame)
            distance_est = self.distance_estimator.estimate_distance_from_height(vert_height)

            # Optionally, keep track of bounding box for debugging or re-identification.
            all_x = [l_sh.x * w, r_sh.x * w, l_hip.x * w, r_hip.x * w]
            all_y = [l_sh.y * h, r_sh.y * h, l_hip.y * h, r_hip.y * h]
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))
            new_bb_w = x_max - x_min
            new_bb_h = y_max - y_min
            
            new_bbox = (x_min, y_min, new_bb_w, new_bb_h)

            if self.visual_tracker.initialized:
                self.visual_tracker.prev_bbox = new_bbox

            wl, wr, angle_int = self.motion_controller.compute_wheel_velocities(
                distance_est,
                user_center_x,
                w,
                self.max_speed
            )

            return wl, wr, distance_est, angle_int, True, new_bbox
        else:
            return 0, 0, None, 0, False, None

    def _check_ultrasonic_emergency_stop(self):
        """
        Checks the ultrasonic sensors to see if any distance is below the configured emergency stop distance.
        Returns True if an obstacle is detected, otherwise False.
        """
        if not self.ultrasonic:
            return False
        distances = self.ultrasonic.get_all_distances()
        # The sensor returns distances in cm. Convert emergency_stop_distance to cm.
        stop_threshold_cm = self.emergency_stop_distance * 100.0
        for sensor_name, dist_cm in distances.items():
            if dist_cm is not None and dist_cm < stop_threshold_cm:
                print(f"[Emergency Stop] Obstacle at {dist_cm} cm on sensor '{sensor_name}'")
                self.request_stop()
                return True
        return False

    def run_tracking_loop(self):
        """
        Main tracking loop. Continuously fetches frames and tries:
         - Pose detection.
         - SIFT-based re-identification if pose is lost.
         - Sends velocity commands.
        Exit if a stop is requested or user presses 'q'.
        """
        print("[Debug] run_tracking_loop() entered.")
        self.send_status_update("Tracking Started")
        last_time_polled = time.time()

        while True:
            # Check if user requested to stop.
            if self.pending_stop:
                self.send_status_update("Tracking Stopped")
                if self.serial_enabled and self.serial_sender:
                    self.serial_sender.send_wheel_velocities(0, 0)
                print("[Debug] Stopping tracking loop.")
                break

            # Check ultrasonic for emergency.
            if self.ultrasonic_enabled:
                if self._check_ultrasonic_emergency_stop():
                    # We just performed a request_stop(), so break out.
                    break

            frame = self.camera_manager.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if self.debug_keys:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.send_status_update("Tracking Stopped")
                    if self.serial_enabled and self.serial_sender:
                        self.serial_sender.send_wheel_velocities(0, 0)
                    print("[Debug] Stopping tracking loop via key.")
                    break

            now = time.time()
            if now - last_time_polled < self.polling_interval:
                continue
            last_time_polled = now

            wl, wr = 0, 0
            distance_est = None
            angle_int = 0
            # for bounding box visualization
            pose_bbox = None
            sift_bbox = None

            # Decide tracking approach based on current state.
            if self.tracking_state == "LOST":
                # Attempt re-ID with SIFT
                print("[Debug] Tracking state is LOST. Trying SIFT re-ID.")
                if self.visual_tracker.initialized:
                    tracked_bbox = self.visual_tracker.track_object_in_frame(frame)
                    if tracked_bbox is not None:
                        # attempt pose again
                        wl, wr, distance_est, angle_int, pose_ok, pose_bbox = self._attempt_pose(
                            frame
                        )
                        if pose_ok:
                            self.tracking_state = "TRACKING"
                            print("[Debug] Pose found after re-ID, switching state to TRACKING.")
                        else:
                            print("[Debug] Pose still not found, remain LOST.")
                            wl, wr = 0, 0
                        # Also record sift_bbox for visualization
                        sift_bbox = tracked_bbox
                    else:
                        print("[Debug] SIFT re-ID failed.")
                        wl, wr = 0, 0
                else:
                    print("[Debug] SIFT not initialized, remain LOST.")
                    wl, wr = 0, 0
            else:
                # Attempt pose detection.
                wl, wr, distance_est, angle_int, pose_ok, pose_bbox = self._attempt_pose(frame)
                if not pose_ok:
                    print("[Debug] Pose detection failed, switching to LOST.")
                    self.tracking_state = "LOST"
                    wl, wr = 0, 0

            print(f"[Tracking Info] state={self.tracking_state}, distance={distance_est}, angle={angle_int} deg, wl={wl:.2f}, wr={wr:.2f}")

            # Send velocities if serial is enabled.
            if self.serial_enabled and self.serial_sender:
                self.serial_sender.send_wheel_velocities(wl, wr)

            if self.draw_enabled and frame is not None:
                draw_frame = frame
                # if we have a pose bounding box, draw it in green
                if pose_bbox is not None:
                    (px, py, pw, ph) = pose_bbox
                    cv2.rectangle(draw_frame, (px, py), (px + pw, py + ph), (0, 255, 0), 2)
                    cv2.putText(draw_frame, "Pose BBox", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # if we have a SIFT bounding box, draw it in red
                if sift_bbox is not None:
                    (sx, sy, sw, sh) = sift_bbox
                    cv2.rectangle(draw_frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
                    cv2.putText(draw_frame, "SIFT BBox", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                cv2.putText(draw_frame, f"State: {self.tracking_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                if distance_est:
                    cv2.putText(draw_frame, f"Dist: {distance_est}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                cv2.putText(draw_frame, f"Angle: {angle_int} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

                cv2.imshow(self.window_name, draw_frame)
                if self.debug_keys:
                    cv2.waitKey(1)

        print("[Debug] Exiting run_tracking_loop.")

    def cleanup_resources(self):
        """
        Stop capturing, close pose, close serial, cleanup GPIO, and destroy any windows.
        This should be called on program exit.
        """
        print("[Debug] Cleaning up resources...")
        if self.serial_enabled and self.serial_sender:
            try:
                self.serial_sender.send_wheel_velocities(0, 0)
                time.sleep(0.05)
            except Exception as e:
                print(f"[Warning] Could not send 0 velocities during cleanup: {e}")
        self.camera_manager.stop_capture()

        if self.pose_estimator and self.pose_estimator.pose:
            try:
                self.pose_estimator.pose.close()
                print("[Debug] Closed MediaPipe Pose in cleanup.")
            except Exception as e:
                print(f"[Warning] Closing pose failed: {e}")

        if self.serial_sender:
            self.serial_sender.close_connection()

        if self.ultrasonic_enabled and self.ultrasonic:
            self.ultrasonic.cleanup_gpio()
        cv2.destroyAllWindows()
        print("[Debug] Cleanup complete.")


def main():
    """
    Entry point of the script. Parses optional arguments, initializes UserTrackerApp,
    starts calibration and the socket listener in a background thread.
    """
    args = sys.argv
    distance_arg = 0.5
    if len(args) > 1:
        try:
            distance_arg = float(args[1])
        except ValueError:
            pass

    tracker = UserTrackerApp(
        camera_index=0,
        target_distance=distance_arg,
        reference_distance=distance_arg,
        polling_interval=0.1,
        serial_port='/dev/ttyUSB0',
        baudrate=9600,
        serial_enabled=False,
        ultrasonic_enabled=False,
        draw_enabled=True,
        debug_keys=True,
        kv=0.3,
        kw=0.005,
        max_speed=0.4,
        emergency_stop_distance=0.25
    )

    #socket listener:
    socket_thread = threading.Thread(target=tracker.command_socket_listener, daemon=True)
    socket_thread.start()

    try:
        tracker.calibrate_distance_reference()
    except KeyboardInterrupt:
        print("[Info] Caught KeyboardInterrupt. Exiting...")
    except Exception as e:
        print(f"[Error] Unhandled exception: {e}")
    finally:
        tracker.cleanup_resources()


if __name__ == "__main__":
    main()