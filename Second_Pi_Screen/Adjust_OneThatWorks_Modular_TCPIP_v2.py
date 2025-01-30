import cv2
import mediapipe as mp
import serial
import sys
# import RPi.GPIO as GPIO
import time
import socket
import numpy as np
import threading
from dataclasses import dataclass, field


# ---------------------------------------------------
#                 CONFIG DATA CLASSES
# ---------------------------------------------------

@dataclass
class CameraConfig:
    """
    Camera Configuration
    """
    index: int = 0        # Camera device index (0 for default built-in or first USB camera)
    width: int = 320      # Width of captured frames
    height: int = 240     # Height of captured frames

@dataclass
class SerialConfig:
    """
    Serial (Robot) Configuration
    """
    enabled: bool = False          # Enable or disable sending velocity commands via serial
    port: str = "/dev/ttyUSB0"     # Serial port device path
    baudrate: int = 9600           # Baud rate for serial communication

@dataclass
class UltrasonicConfig:
    """
    Ultrasonic Sensor Configuration
    """
    enabled: bool = False   # Enable or disable ultrasonic distance measurements

@dataclass
class TcpConfig:
    """
    TCP Communication Configuration
    """
    enabled: bool = False         # Enable or disable TCP client behavior (e.g., sending commands to an LCD Pi)
    host: str = "140.193.235.72"  # Host/IP of the remote machine (LCD Pi)
    port: int = 12345             # TCP port on the remote machine

@dataclass
class DrawConfig:
    """
    Drawing / Visualization Configuration
    """
    enabled: bool = True   # Enable or disable on-screen drawings (debug overlays)

@dataclass
class TrackerConfig:
    """
    Tracker Configuration
    """
    polling_interval: float = 0.0       # How often (in seconds) to run the main tracking logic
    reference_distance: float = 0.5     # Known real-world distance for calibration (meters)
    target_distance: float = 0.5        # Desired distance from user to robot (meters)

@dataclass
class MovementConfig:
    """
    Movement / Controller Configuration
    """
    kv: float = 0.27                            # Linear velocity gain
    kw: float = 0.00055                         # Angular velocity gain
    distance_tolerance: float = 0.1             # Distance tolerance for stopping, etc. (not heavily used)
    angle_tolerance_deg: float = 25             # Angle tolerance in degrees (not heavily used)
    angle_smoothing_threshold: float = 2.5      # Angle change threshold for smoothing
    angle_deadband: float = 10.0                # Range that we treat as "straight" (no angular error)
    radius: float = 0.5                         # Effective wheel radius or factor for converting velocity to wheel speed
    max_wheel_speed: float = 0.1                # Upper bound for wheel speed
    min_wheel_speed: float = -0.1               # Lower bound for wheel speed

@dataclass
class AppConfig:
    """
    Top-Level App Config containing all subconfigs.
    """
    camera: CameraConfig = field(default_factory=CameraConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    ultrasonic: UltrasonicConfig = field(default_factory=UltrasonicConfig)
    tcp: TcpConfig = field(default_factory=TcpConfig)
    draw: DrawConfig = field(default_factory=DrawConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)


# ---------------------------------------------------
#             SIFT-BASED FEATURE TRACKER
# ---------------------------------------------------
class FeatureTracker:
    def __init__(self):
        # Create SIFT and FLANN
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Internals
        self.object_descriptors = None
        self.background_descriptors = None
        self.prev_bbox = None
        self.initialized = False

    def initialize_tracking(self, frame, bbox):
        """
        Initialize the SIFT-based tracker with an ROI specified by bbox.
        bbox = (x, y, w, h)
        """
        self.prev_bbox = bbox

        x, y, w, h = bbox
        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None:
            return

        object_keypoints = []
        object_descriptors = []
        background_keypoints = []
        background_descriptors = []

        for kp, desc in zip(keypoints, descriptors):
            if (x <= kp.pt[0] <= x + w) and (y <= kp.pt[1] <= y + h):
                object_keypoints.append(kp)
                object_descriptors.append(desc)
            else:
                background_keypoints.append(kp)
                background_descriptors.append(desc)

        if len(object_descriptors) > 0:
            self.object_descriptors = np.array(object_descriptors, dtype=np.float32)
        else:
            self.object_descriptors = None

        if len(background_descriptors) > 0:
            self.background_descriptors = np.array(background_descriptors, dtype=np.float32)
        else:
            self.background_descriptors = None

        self.initialized = True

    def track_object(self, frame):
        """
        Uses SIFT matching + FLANN to update the bounding box.
        Returns the new_bbox (x, y, w, h) or None if tracking fails.
        """
        if (not self.initialized or
            self.object_descriptors is None or
            self.background_descriptors is None):
            return None

        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None or len(keypoints) == 0:
            return None

        try:
            object_matches = self.flann.knnMatch(descriptors, self.object_descriptors, k=2)
            background_matches = self.flann.knnMatch(descriptors, self.background_descriptors, k=2)
        except:
            return None

        object_points = []
        for i, kp in enumerate(keypoints):
            if i < len(object_matches) and len(object_matches[i]) == 2:
                # Best match distance for object
                d_o1 = object_matches[i][0].distance
                # For background
                if i < len(background_matches) and len(background_matches[i]) > 0:
                    d_b1 = background_matches[i][0].distance
                else:
                    d_b1 = float('inf')

                if d_b1 == 0:  # avoid division by zero
                    continue

                ratio = d_o1 / d_b1
                if ratio < 0.5:  # accept match
                    object_points.append(kp.pt)

        if len(object_points) == 0:
            return None

        points = np.array(object_points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        new_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        self.prev_bbox = new_bbox
        return new_bbox

# ---------------------------------------------------
#                CAMERA HANDLER
# ---------------------------------------------------
class CameraHandler:
    """
    Camera Input Handler

    This class initializes and provides frames from a webcam or
    other camera source. Frame resolution is set lower to reduce
    computational load.
    """

    def __init__(self, camera_index=0, width=320, height=240):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        """
        Returns the current frame from the camera.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """
        Releases the camera device.
        """
        self.cap.release()

# ---------------------------------------------------
#         MEDIAPIPE POSE ESTIMATION
# ---------------------------------------------------
class PoseEstimation:
    """
    Pose Estimation

    This class uses MediaPipe's Pose solution to detect key landmarks.
    It extracts relevant positions for distance and angle calculations.
    """

    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        """
        Processes a given BGR frame, returns pose landmarks if found.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

# ---------------------------------------------------
#            ULTRASONIC SENSOR
# ---------------------------------------------------
class UltrasonicSystem:
    def __init__(self):
        sensors, buffer_size, measurements, timeout = self._setup_gpio_sensors()
        self.sensors = {name: {"TRIG": config["TRIG"], "ECHO": config["ECHO"]} for name, config in sensors.items()}
        self.distance_buffers = {name: [] for name in sensors}
        self.cached_distances = {name: None for name in sensors}
        self.lock = threading.Lock()
        self.TIMEOUT = timeout
        self.BUFFER_SIZE = buffer_size
        self.MEASUREMENTS = measurements

    def _setup_gpio_sensors(self):
        """
        Private method that performs GPIO initialization for ultrasonic sensors.
        """
        SENSORS = {
            "front": {"TRIG": 11, "ECHO": 12},
            "left": {"TRIG": 13, "ECHO": 16},
        }
        BUFFER_SIZE = 5
        MEASUREMENTS = 5
        TIMEOUT = 0.04  # 40ms timeout

        # If on a real Pi, uncomment:
        # GPIO.setmode(GPIO.BCM)
        # for sensor in SENSORS.values():
        #     GPIO.setup(sensor["TRIG"], GPIO.OUT)
        #     GPIO.setup(sensor["ECHO"], GPIO.IN)

        return SENSORS, BUFFER_SIZE, MEASUREMENTS, TIMEOUT

    def measure_distance(self, trig_pin, echo_pin):
        # GPIO.output(trig_pin, True)
        time.sleep(0.00001)  # 10 microseconds
        # GPIO.output(trig_pin, False)

        start_time = time.time()
        timeout_time = start_time + self.TIMEOUT

        # Example of a real loop:
        # while GPIO.input(echo_pin) == 0:
        #     if time.time() > timeout_time:
        #         return None
        #     start_time = time.time()

        # while GPIO.input(echo_pin) == 1:
        #     if time.time() > timeout_time:
        #         return None
        stop_time = time.time()

        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2

        return distance if 2 <= distance <= 500 else None

    def async_measure(self, name, trig_pin, echo_pin):
        dist = self.measure_distance(trig_pin, echo_pin)
        if dist is not None:
            with self.lock:
                self.cached_distances[name] = dist

    def read_distances(self):
        threads = []
        for name, pins in self.sensors.items():
            thread = threading.Thread(target=self.async_measure, args=(name, pins['TRIG'], pins['ECHO']))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.cached_distances

    def cleanup(self):
        # GPIO.cleanup()
        pass

# ---------------------------------------------------
#         DISTANCE CALCULATOR (CALIBRATION)
# ---------------------------------------------------
class DistanceEstimator:
    """
    Distance Calculator

    This class uses a reference height (user's shoulder-to-hip in the frame)
    at a known distance to estimate the user's current distance.
    """

    def __init__(self, reference_distance=0.5):
        self.reference_distance = reference_distance  # meters
        self.reference_height = None                 # pixels (shoulder-to-hip at reference distance)

    def calibrate(self, vertical_height):
        """
        Sets the reference height based on a known distance.
        """
        self.reference_height = vertical_height
        print(f"[CALIBRATION] Reference height set: {vertical_height:.2f} pixels "
              f"for {self.reference_distance} meters.")

    def estimate_distance(self, current_height):
        """
        Returns an estimated distance in meters based on the ratio
        of current_height to the reference_height.
        """
        if self.reference_height is not None and current_height > 0:
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None

# ---------------------------------------------------
#      MOVEMENT CONTROLLER (Velocity Calc)
# ---------------------------------------------------
class MovementController:
    """
    Movement Controller

    Calculates linear and angular velocities given distance and angle errors.
    """

    def __init__(self, movement_config: MovementConfig, target_distance):
        """
        movement_config : MovementConfig dataclass containing gains, thresholds, etc.
        target_distance : float, desired distance to maintain from the user
        """
        self.movement_config = movement_config
        self.target_distance = target_distance
        self.distance_tolerance = movement_config.distance_tolerance
        self.angle_tolerance_deg = movement_config.angle_tolerance_deg
        self.kv = movement_config.kv
        self.kw = movement_config.kw

    def compute_control(self, distance, angle_offset_deg):
        """
        Given the current distance (meters) and angle offset (degrees),
        returns the linear and angular velocities to send to the robot.
        """
        distance_error = 0 if distance is None else (distance - self.target_distance)
        angle_error_deg = angle_offset_deg

        linear_vel = self.kv * distance_error
        angular_vel = self.kw * angle_error_deg

        return linear_vel, angular_vel

# ---------------------------------------------------
#              SERIAL OUTPUT (Robot)
# ---------------------------------------------------
class SerialOutput:
    """
    Serial Output

    Sends commands (in this case, linear and angular velocities) to
    the robot over a serial connection.
    """

    def __init__(self, port, baudrate=9600):
        self.ser = serial.Serial(port, baudrate)
        return

    def send_velocities(self, wl, wr):
        """
        Sends the linear and angular velocities over serial
        as a formatted string (e.g., "w_l:0.1 w_r:3.5").
        """
        msg = f"w_l:{wl:.1f} w_r:{wr:.1f}\n"
        self.ser.write(msg.encode('utf-8'))
        return

    def close(self):
        """
        Closes the serial connection.
        """
        self.ser.close()

# ---------------------------------------------------
#      MAIN TRACKER ORCHESTRATION
# ---------------------------------------------------
class TrackerOrchestration:
    """
    Main class that orchestrates everything:
      1. Capture frames
      2. Detect pose (shoulder & hip positions)
      3. Estimate distance
      4. Compute linear & angular velocities
      5. Send to serial output
      6. (Optionally) read from ultrasonic sensors
      7. (Optionally) send commands to LCD Pi
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize nodes
        print("Initializing Camera")
        self.camera = CameraHandler(
            camera_index=config.camera.index,
            width=config.camera.width,
            height=config.camera.height
        )
        print("Initializing Pose Estimation")
        self.pose_estimation = PoseEstimation()
        print("Initializing Distance Calculator")
        self.distance_calculator = DistanceEstimator(
            reference_distance=config.tracker.reference_distance
        )
        print("Initializing Controller")
        self.movement_controller = MovementController(
            movement_config=config.movement,
            target_distance=config.tracker.target_distance
        )

        # Toggles
        self.serial_enabled = config.serial.enabled
        self.ultrasonic_enabled = config.ultrasonic.enabled
        self.TCP_enabled = config.tcp.enabled
        self.draw_enabled = config.draw.enabled

        if self.serial_enabled:
            print("Initializing Serial Output")
            self.serial_output = SerialOutput(
                port=config.serial.port,
                baudrate=config.serial.baudrate
            )

        if self.ultrasonic_enabled:
            print("Initializing Ultrasonic System")
            self.ultrasonic_system = UltrasonicSystem()

        print("Initializing Feature-Based Tracker")
        self.feature_tracker = FeatureTracker()

        self.sift_bbox = None
        self.window_name = "Distance & Angle Tracker"

        self.polling_interval = config.tracker.polling_interval
        self.last_poll_time = time.time()

        # For angle smoothing
        self.angle_smoothing_threshold = config.movement.angle_smoothing_threshold
        self.angle_deadband = config.movement.angle_deadband
        self.previous_angle_offset_int = 0

    def _send_command_tcp(self, command):
        """
        Sends a command via TCP to the LCD Pi server.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.config.tcp.host, self.config.tcp.port))
                msg = command.strip() + "\n"
                s.sendall(msg.encode("utf-8"))
        except Exception as e:
            print(f"[TCP] Error sending '{command}' to {self.config.tcp.host}:{self.config.tcp.port} -> {e}")

    def signal_status(self, status):
        """
        Optionally send status to a local UNIX socket (if available).
        """
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect("/tmp/status_socket")
                sock.sendall(status.encode())
        except FileNotFoundError:
            pass  # Socket not found; ignore

    def _analyze_pose(self, frame):
        """
        Detects pose in the given frame using MediaPipe.
        Returns:
            - vertical_height (float): the vertical height (shoulder-hip)
            - user_center_x (float): average x for the shoulders
            - user_center_y (float): average y for the shoulders
            - landmarks (list): raw pose landmarks
            - pose_found (bool): whether pose was found
        """
        results = self.pose_estimation.process_frame(frame)
        if not results.pose_landmarks:
            return 0, 0, 0, None, False

        landmarks = results.pose_landmarks.landmark
        frame_height, frame_width = frame.shape[:2]

        # Shoulders
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        # Hips
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
        hip_y = (left_hip.y + right_hip.y) / 2 * frame_height
        vertical_height = abs(shoulder_y - hip_y)

        user_center_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_width
        user_center_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_height

        return vertical_height, user_center_x, user_center_y, landmarks, True

    def _manage_sift_bbox(self, frame, landmarks, pose_detected):
        """
        Initialize or update SIFT bounding box based on mediapipe
        shoulder and hip landmarks. If pose not detected, tries
        to track object via SIFT.
        """
        if pose_detected:
            frame_height, frame_width = frame.shape[:2]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

            all_x = [
                left_shoulder.x * frame_width,
                right_shoulder.x * frame_width,
                left_hip.x * frame_width,
                right_hip.x * frame_width
            ]
            all_y = [
                left_shoulder.y * frame_height,
                right_shoulder.y * frame_height,
                left_hip.y * frame_height,
                right_hip.y * frame_height
            ]
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))
            w = x_max - x_min
            h = y_max - y_min

            # Re-init each loop so bounding box tracks the user
            self.feature_tracker.initialize_tracking(frame, (x_min, y_min, w, h))
            self.sift_bbox = (x_min, y_min, w, h)

        else:
            new_bbox = self.feature_tracker.track_object(frame)
            if new_bbox is not None:
                self.sift_bbox = new_bbox
            else:
                cv2.putText(frame, "Tracking lost!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                if self.serial_enabled:
                    self.serial_output.send_velocities(0, 0)

    def _overlay_info(self, frame, distance, linear_vel, angular_vel, angle_offset_deg,
                      user_center_x, user_center_y, bbox=None, pose_detected=False):
        """
        Draw textual info + bounding box + a small line for torso center if pose is detected.
        """
        text_color = (0, 255, 255)
        font_scale = 0.5

        cv2.putText(frame, f"Distance: {distance if distance else 0:.2f} m",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Linear Vel: {linear_vel:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Angular Vel: {angular_vel:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Angle Offset: {angle_offset_deg:.2f} deg",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"User Ctr: ({int(user_center_x)}, {int(user_center_y)})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        if bbox is not None:
            (bx, by, bw, bh) = bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        if pose_detected:
            cx = int(user_center_x)
            cy = int(user_center_y)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)

    # ------------------------------------------------------
    #                   PUBLIC METHODS
    # ------------------------------------------------------
    def calibrate_distance(self):
        """
        Allows the user to calibrate the reference height by pressing 'c'.
        Press 'q' to exit early.

        """
        print("Stand at the known distance and press 'c' to calibrate reference height (or 'q' to quit).")
        sentStatus = False

        while True:
            frame = self.camera.get_frame()
            if frame is None:
                print("Could not access the camera.")
                break

            vertical_height, sh_center_x, sh_center_y, landmarks, pose_found = \
                self._analyze_pose(frame)

            if pose_found and landmarks is not None:
                # Shoulders & hips from landmarks:
                left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                # Convert from normalized to pixel coords:
                sh_x = int((left_shoulder.x + right_shoulder.x)/2 * frame.shape[1])
                sh_y = int((left_shoulder.y + right_shoulder.y)/2 * frame.shape[0])
                hp_x = int((left_hip.x + right_hip.x)/2 * frame.shape[1])
                hp_y = int((left_hip.y + right_hip.y)/2 * frame.shape[0])

                # Draw the actual shoulder–hip line
                cv2.line(frame, (sh_x, sh_y), (hp_x, hp_y), (0, 255, 0), 2)
                cv2.putText(frame, "Calibrate Height: Press 'c' to set",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                cv2.imshow("Calibrate Reference Distance", frame)

                if not sentStatus:
                    self.signal_status("Calibration Started")
                    sentStatus = True

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.distance_calculator.calibrate(vertical_height)
                    self.run_main_loop()  # calls the main loop
                    break
                elif key == ord('q'):
                    self.signal_status("Calibration Stopped")
                    break
            else:
                cv2.imshow("Calibrate Reference Distance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyWindow("Calibrate Reference Distance")

    def run_main_loop(self, draw_enabled=False, ultrasonic_enabled=False, TCP_enabled=False):
        """
        Main loop that captures frames, estimates distance & angle,
        computes velocities, sends them over serial,
        and also sends commands to LCD Pi via TCP if enabled.

        Press 'q' to quit.
        """
        sentStatus = False
        skip_polling_check = False

        while True:
            frame = self.camera.get_frame()
            if frame is None:
                print("[ERROR] Could not read frame.")
                break

            # If ultrasonic is enabled
            if self.ultrasonic_enabled:
                distances = self.ultrasonic_system.read_distances()
                # Example usage:
                # if any(distance is not None and distance < 30 for distance in distances.values()):
                #     if self.serial_enabled:
                #         self.serial_output.send_velocities(0, 0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.serial_enabled:
                    self.serial_output.send_velocities(0, 0)
                self.signal_status("Tracking Stopped")
                skip_polling_check = True
                break
            elif key == ord('c'):
                # re-calibrate distance
                self.signal_status("Calibration")
                self.calibrate_distance()

            current_time = time.time()
            if not skip_polling_check and (current_time - self.last_poll_time) >= self.polling_interval:
                self.last_poll_time = current_time

                frame_height, frame_width = frame.shape[:2]
                vertical_height, user_center_x, user_center_y, landmarks, pose_detected = \
                    self._analyze_pose(frame)

                # Default
                distance = 0
                linear_vel = 0
                angular_vel = 0
                wl = 0
                wr = 0

                if not sentStatus:
                    self.signal_status("Tracking Started")
                    sentStatus = True

                if pose_detected:
                    # Estimate distance from user
                    distance = self.distance_calculator.estimate_distance(vertical_height)

                    # --- This block EXACTLY matches your “second script” logic & prints ---
                    horizontal_offset = user_center_x - (frame_width / 2)
                    normalized_offset = horizontal_offset / (frame_width / 2)
                    angle_offset_deg = max(min(normalized_offset * 90, 90), -90)
                    print("actual angle:", angle_offset_deg)

                    # Smooth the angle offset into an int
                    angle_offset_int = int(angle_offset_deg)
                    if abs(angle_offset_int - self.previous_angle_offset_int) <= self.angle_smoothing_threshold:
                        angle_offset_int = self.previous_angle_offset_int
                    else:
                        self.previous_angle_offset_int = angle_offset_int

                    # Deadband
                    if -self.angle_deadband <= angle_offset_int <= self.angle_deadband:
                        angle_offset_int = 0

                    # Compute velocities via your MovementController
                    linear_vel, angular_vel = self.movement_controller.compute_control(distance, angle_offset_int)

                    # Convert those velocities to left/right wheel speeds
                    radius = self.config.movement.radius
                    w_hat_l = linear_vel / radius
                    w_hat_r = w_hat_l
                    wl = angular_vel + w_hat_l
                    wr = -angular_vel + w_hat_r

                    # Clip to max/min
                    max_speed = self.config.movement.max_wheel_speed
                    min_speed = self.config.movement.min_wheel_speed
                    wr = max(min_speed, min(wr, max_speed))
                    wl = max(min_speed, min(wl, max_speed))

                    # Print exactly like the second script
                    print("Angular Velocity:", angular_vel)
                    print("Angle Offset:", angle_offset_int)
                    print("wr:", wr, " ", "wl:", wl)

                    # Send to serial if enabled
                    if self.serial_enabled:
                        self.serial_output.send_velocities(wl, wr)

                    # Optionally send commands via TCP
                    if self.TCP_enabled:
                        if angle_offset_int < -self.angle_deadband:
                            self._send_command_tcp("left")
                        elif angle_offset_int > self.angle_deadband:
                            self._send_command_tcp("right")
                        else:
                            self._send_command_tcp("straight")

                # SIFT bounding box update
                self._manage_sift_bbox(frame, landmarks, pose_detected)

                # If draw is enabled, overlay everything
                if self.draw_enabled:
                    self._overlay_info(
                        frame=frame,
                        distance=distance,
                        linear_vel=linear_vel,
                        angular_vel=angular_vel,
                        angle_offset_deg=angle_offset_int,  # show final int offset
                        user_center_x=user_center_x,
                        user_center_y=user_center_y,
                        bbox=self.sift_bbox,
                        pose_detected=pose_detected
                    )

            if self.draw_enabled:
                cv2.imshow(self.window_name, frame)

        # Cleanup
        self.camera.release()
        if self.serial_enabled:
            self.serial_output.close()
        if self.ultrasonic_enabled:
            self.ultrasonic_system.cleanup()
        # cv2.destroyAllWindows()


# ----------------------------
#             MAIN
# ----------------------------
if __name__ == "__main__":
    # Create a default AppConfig (all parameters are stored in dataclasses)
    config = AppConfig()

    # Optional command-line argument for target distance
    args = sys.argv
    if len(args) > 1:
        distance_arg = float(args[1])
    else:
        distance_arg = 0.5

    if not (0 < distance_arg < 1):
        distance_arg = 0.5

    # Overwrite target distance from CLI
    config.tracker.target_distance = distance_arg

    # Instantiate main tracker class
    tracker = TrackerOrchestration(config)

    # Calibrate reference height, which then calls run_main_loop()
    tracker.calibrate_distance()

    print("Ending Script")


