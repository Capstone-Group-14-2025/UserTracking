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

# If you have Raspberry Pi and real ultrasonic sensors, uncomment and configure:
# import RPi.GPIO as GPIO
# import lgpio

# ----------------------------------------
#          CONFIG + CONSTANTS
# ----------------------------------------
SENSORS = {
    "front": {"TRIG": 11, "ECHO": 12},
    "left":  {"TRIG": 13, "ECHO": 16},
    # "right": {"TRIG": 17, "ECHO": 18},
    # "back":  {"TRIG": 19, "ECHO": 20},
}
TIMEOUT = 0.04  # 40ms for ultrasonic timeouts

# Path for listening to commands from Bluetooth code
COMMAND_SOCKET_PATH = "/tmp/command_socket"


# ----------------------------------------
#    SOCKET LISTENER THREAD
# ----------------------------------------
def listen_for_socket_commands(tracker=None):
    """
    Listens for commands on /tmp/command_socket in a background thread.
    Recognized commands:
      - 'stop'        => set tracker.pending_stop = True
      - 'calibrate'   => set tracker.pending_calibrate = True
      - 'start:0.8'   => set tracker.pending_start = True AND set target distance to 0.8
    """
    if os.path.exists(COMMAND_SOCKET_PATH):
        os.remove(COMMAND_SOCKET_PATH)
    
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_sock:
        server_sock.bind(COMMAND_SOCKET_PATH)
        server_sock.listen(1)
        print(f"[Control.py] Listening for external commands on {COMMAND_SOCKET_PATH}")
        
        while True:
            conn, _ = server_sock.accept()
            with conn:
                data = conn.recv(1024).decode().strip().lower()
                if not data:
                    print("Con:",conn, "------------------------------------------------")
                    continue

                parts = data.split(':')
                cmd = parts[0]

                if cmd == "stop":
                    print("[Socket] STOP command received.")
                    if tracker is not None:
                        tracker.set_stop()

                elif cmd == "calibrate":
                    print("[Socket] CALIBRATE command received.")
                    if tracker is not None:
                        tracker.set_calibrate()

                elif cmd == "start":
                    # If "start:<distance>" is provided, parse that new distance
                    distance_val = None
                    if len(parts) > 1:
                        try:
                            distance_val = float(parts[1])
                        except ValueError:
                            print(f"[Socket] Invalid distance in command: {data}")

                    if tracker is not None:
                        tracker.set_start(distance_val)
                else:
                    print(f"[Socket] Unknown command: {data}")

            print("Con:",conn, "------------------------------------------------")


# ----------------------------------------
#      FEATURE-BASED SIFT TRACKER
# ----------------------------------------
class FeatureBasedTracker:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.object_descriptors = None
        self.background_descriptors = None
        self.prev_bbox = None
        self.initialized = False

    def initialize_tracking(self, frame, bbox):
        """
        Initialize the SIFT-based tracker with an ROI specified by bbox = (x, y, w, h).
        """
        self.prev_bbox = bbox
        x, y, w, h = bbox

        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None:
            return

        obj_desc_list = []
        bg_desc_list = []

        for kp, desc in zip(keypoints, descriptors):
            if (x <= kp.pt[0] <= x + w) and (y <= kp.pt[1] <= y + h):
                obj_desc_list.append(desc)
            else:
                bg_desc_list.append(desc)

        if len(obj_desc_list) > 0:
            self.object_descriptors = np.array(obj_desc_list, dtype=np.float32)
        else:
            self.object_descriptors = None

        if len(bg_desc_list) > 0:
            self.background_descriptors = np.array(bg_desc_list, dtype=np.float32)
        else:
            self.background_descriptors = None

        self.initialized = True

    def track_object(self, frame):
        """
        Uses SIFT matching + FLANN to update the bounding box.
        Returns new_bbox = (x, y, w, h) or None if tracking fails.
        """
        if (not self.initialized or
            self.object_descriptors is None or
            self.background_descriptors is None):
            return None

        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None or len(keypoints) == 0:
            return None

        # KNN match
        try:
            object_matches = self.flann.knnMatch(descriptors, self.object_descriptors, k=2)
            background_matches = self.flann.knnMatch(descriptors, self.background_descriptors, k=2)
        except:
            return None

        object_points = []
        for i, kp in enumerate(keypoints):
            if i < len(object_matches) and len(object_matches[i]) == 2:
                d_o1 = object_matches[i][0].distance
                if i < len(background_matches) and len(background_matches[i]) > 0:
                    d_b1 = background_matches[i][0].distance
                else:
                    d_b1 = float('inf')

                if d_b1 == 0:
                    continue
                ratio = d_o1 / d_b1
                if ratio < 0.5:
                    object_points.append(kp.pt)

        if len(object_points) == 0:
            return None

        pts = np.array(object_points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        new_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        self.prev_bbox = new_bbox
        return new_bbox


# ----------------------------------------
#            CAMERA NODE
# ----------------------------------------
class CameraNode:
    """
    Runs a separate thread to continually capture frames from the camera.
    """
    def __init__(self, camera_index=0, width=320, height=240, max_retries=5, delay=1):
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

        self.release_camera_if_busy()
        self.initialize_camera()

    def release_camera_if_busy(self):
        """
        Linux-only approach to free /dev/video0 from other processes.
        Remove/disable on non-Linux or if not needed.
        """
        cam_process = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
        if cam_process.stdout.strip():
            print(f"Camera is locked by {cam_process.stdout.strip()}, attempting to free it.")
            os.system("sudo fuser -k /dev/video0")
            time.sleep(1)
        else:
            print("Camera not in use.")

    def initialize_camera(self):
        for attempt in range(1, self.max_retries + 1):
            print(f"Attempting to open camera (Try {attempt}/{self.max_retries})...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print("Camera opened successfully.")
                return
            print("Camera open failed. Retrying...")
            self.release_camera_if_busy()
            time.sleep(self.delay)

        print("Failed to open camera after multiple attempts.")
        self.cap = None

    def start(self):
        if self.cap is None:
            print("No camera available to start.")
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def retrieve_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
            print("Camera released.")


# ----------------------------------------
#         MEDIAPIPE POSE NODE
# ----------------------------------------
class PoseEstimationNode:
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)


# ----------------------------------------
#         ULTRASONIC SENSOR
# ----------------------------------------
class UltrasonicSensor:
    def __init__(self):
        self.sensors = {name: {"TRIG": cfg["TRIG"], "ECHO": cfg["ECHO"]} for name, cfg in SENSORS.items()}
        self.cached_distances = {name: None for name in self.sensors}
        self.lock = threading.Lock()

    def measure_distance(self, trig, echo):
        """
        Dummy function. Replace with real code if using actual hardware.
        """
        # Example with lgpio or RPi.GPIO goes here.
        # Return distance in cm or None if invalid.
        return 0  # for demonstration

    def _thread_measure(self, sensor_name, trig, echo):
        dist = self.measure_distance(trig, echo)
        with self.lock:
            self.cached_distances[sensor_name] = dist

    def read_distances(self):
        threads = []
        for name, pins in self.sensors.items():
            t = threading.Thread(target=self._thread_measure, args=(name, pins["TRIG"], pins["ECHO"]))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return self.cached_distances

    def cleanup(self):
        # If you have GPIO or lgpio open, clean it up here
        pass


# ----------------------------------------
#      DISTANCE CALCULATOR NODE
# ----------------------------------------
class DistanceCalculator:
    def __init__(self, reference_distance=0.5):
        self.reference_distance = reference_distance  # known distance in meters
        self.reference_height = None                 # in pixels

    def calibrate(self, vertical_height):
        self.reference_height = vertical_height
        print(f"Calibrated reference height = {vertical_height:.2f} px at distance = {self.reference_distance} m")

    def estimate_distance(self, current_height):
        if self.reference_height is not None and current_height > 0:
            # Distance ~ reference_distance * (reference_height / current_height)
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None


# ----------------------------------------
#   MOVEMENT CONTROLLER (LOGIC ONLY)
# ----------------------------------------
class MovementController:
    def __init__(self,
                 kv=0.1,
                 kw=0.001,
                 target_distance=0.5,
                 distance_tolerance=0.1,
                 angle_tolerance_deg=25):
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        self.kv = kv  # linear gain
        self.kw = kw  # angular gain

    def compute_control(self, distance, angle_offset_deg):
        if distance is None:
            distance_err = 0
        else:
            distance_err = distance - self.target_distance

        angle_err = angle_offset_deg

        linear_vel = self.kv * distance_err
        angular_vel = self.kw * angle_err

        return linear_vel, angular_vel


# ----------------------------------------
#       SERIAL OUTPUT NODE
# ----------------------------------------
class SerialOutput:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        self.ser = serial.Serial(port, baudrate)

    def send_velocities(self, wl, wr):
        msg = f"w_l:{wl:.3f} w_r:{wr:.3f}\n"
        self.ser.write(msg.encode('utf-8'))

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ----------------------------------------
#  MAIN DISTANCE & ANGLE TRACKER CLASS
# ----------------------------------------
class DistanceAngleTracker:
    def __init__(self,
                 camera_index=0,
                 target_distance=0.5,
                 reference_distance=0.5,
                 polling_interval=0.3,
                 serial_port='/dev/ttyUSB0',
                 baudrate=9600,
                 serial_enabled=False,
                 draw_enabled=True,
                 kv=0.1,
                 kw=0.00095):
        # Settings
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()
        self.serial_enabled = serial_enabled
        self.draw_enabled = draw_enabled
        self.debug_keys = False  # <--- If True, we allow emergency keyboard usage

        # Command flags set by the socket listener
        self.pending_start = False
        self.pending_stop = False
        self.pending_calibrate = False

        # Initialize modules
        print("Initializing camera node...")
        self.camera_node = CameraNode(camera_index=camera_index)
        self.camera_node.start()

        print("Initializing pose estimation node...")
        self.pose_node = PoseEstimationNode()

        print("Initializing distance calculator...")
        self.distance_calculator = DistanceCalculator(reference_distance)

        print("Initializing movement controller...")
        self.movement_controller = MovementController(kv=kv, kw=kw, target_distance=target_distance)

        if self.serial_enabled:
            print("Initializing serial output...")
            self.serial_output = SerialOutput(port=serial_port, baudrate=baudrate)
        else:
            self.serial_output = None

        print("Initializing ultrasonic sensor node...")
        self.ultrasonic_sensor = UltrasonicSensor()

        print("Initializing feature-based tracker...")
        self.feature_tracker = FeatureBasedTracker()

        # For drawing
        self.window_name = "Distance & Angle Tracker"

    # --------------------------------------------------
    #  Methods called by socket commands
    # --------------------------------------------------
    def set_stop(self):
        self.pending_stop = True

    def set_calibrate(self):
        self.pending_calibrate = True

    def set_start(self, distance=None):
        # If a distance is provided, update the target distance
        if distance is not None:
            self.movement_controller.target_distance = distance
            self.distance_calculator.reference_distance = distance
        self.pending_start = True

    # --------------------------------------------------
    #  Optional: send status to /tmp/status_socket
    # --------------------------------------------------
    def signal_status(self, status_message):
        """
        Example method to signal status over a UNIX socket. 
        Will do nothing if the socket doesn't exist.
        """
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect("/tmp/status_socket")
                sock.sendall(status_message.encode())
        except FileNotFoundError:
            pass

    # --------------------------------------------------
    #            Calibration Method
    # --------------------------------------------------
    def calibrate_reference(self):
        """
        We no longer wait for 'c' or 'q' from the keyboard.
        Instead, we:
          - loop, checking if we see a pose
          - if we do, calibrate automatically
          - if pending_stop is set, we exit calibration
        """
        self.signal_status("Calibration Started")
        while True:
            # If "stop" was commanded, break immediately
            if self.pending_stop:
                print("[Calibration] Stop requested.")
                self.signal_status("Calibration Stopped")
                break

            frame = self.camera_node.retrieve_latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # Optional debug: press 'c' or 'q' only if debug_keys is True
            if self.debug_keys:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    pass  # We'll let the auto-calibration run
                elif key == ord('q'):
                    self.signal_status("Calibration Stopped")
                    break

            results = self.pose_node.process_frame(frame)
            if results and results.pose_landmarks:
                # Once we see a valid pose, calibrate automatically
                landmarks = results.pose_landmarks.landmark
                h, w = frame.shape[:2]

                # Shoulders
                l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                # Hips
                l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                shoulder_y = (l_sh.y + r_sh.y) / 2 * h
                hip_y = (l_hip.y + r_hip.y) / 2 * h
                vert_height = abs(shoulder_y - hip_y)

                self.distance_calculator.calibrate(vert_height)
                

                # After calibration, automatically start tracking
                if(self.pending_calibrate):
                    self.start_tracking()
                    break

            # Show for debugging (if draw_enabled)
            if self.draw_enabled:
                debug_frame = frame.copy()
                cv2.putText(debug_frame, "Calibrating... (send 'stop' to abort)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.imshow("Calibrate Reference Distance", debug_frame)
                cv2.waitKey(1)

        cv2.destroyWindow("Calibrate Reference Distance")

    # --------------------------------------------------
    #            Tracking Method
    # --------------------------------------------------
    def start_tracking(self):
        """
        Main loop for distance & angle tracking.
        We no longer rely solely on 'q' from keyboard to stop.
        Instead, we watch for self.pending_stop = True.
        """
        print("=== Start Tracking Mode ===")
        self.signal_status("Tracking Started")
        last_time_polled = time.time()
        previous_angle_offset = 0

        while True:
            # If "stop" was commanded, break out
            if self.pending_stop:
                self.signal_status("Tracking Stopped")
                break 

            frame = self.camera_node.retrieve_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Optional debug: press 'q' if debug_keys is True
            if self.debug_keys:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[Tracking] Stopped via keyboard.")
                    self.signal_status("Tracking Stopped")
                    break

            # Check ultrasonic sensors
            distances = self.ultrasonic_sensor.read_distances()
            # (Optional logic to handle obstacles)

            # Throttle main logic
            now = time.time()
            if now - last_time_polled >= self.polling_interval:
                last_time_polled = now
                results = self.pose_node.process_frame(frame)

                if results and results.pose_landmarks:
                    h, w = frame.shape[:2]
                    landmarks = results.pose_landmarks.landmark

                    # Shoulders
                    l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    # Hips
                    l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                    # Estimate distance
                    shoulder_y = (l_sh.y + r_sh.y) / 2 * h
                    hip_y = (l_hip.y + r_hip.y) / 2 * h
                    vert_height = abs(shoulder_y - hip_y)
                    distance_est = self.distance_calculator.estimate_distance(vert_height)

                    # Compute horizontal angle offset
                    user_center_x = (l_sh.x + r_sh.x) / 2 * w
                    offset = user_center_x - (w / 2)
                    norm_offset = offset / (w / 2)
                    angle_deg = max(min(norm_offset * 90, 90), -90)

                    # Simple smoothing
                    angle_int = int(angle_deg)
                    if abs(angle_int - previous_angle_offset) <= 2:
                        angle_int = previous_angle_offset
                    else:
                        previous_angle_offset = angle_int

                    # Compute wheel speeds
                    linear_vel, angular_vel = self.movement_controller.compute_control(distance_est, angle_int)
                    wheelbase_factor = 0.5
                    base_speed = linear_vel / wheelbase_factor
                    wl = angular_vel + base_speed
                    wr = -angular_vel + base_speed
                    # clamp
                    max_speed = 0.08
                    wl = max(min(wl, max_speed), -max_speed)
                    wr = max(min(wr, max_speed), -max_speed)

                    print(f"[Tracking] Dist={distance_est}, Angle={angle_int}, wl={wl:.3f}, wr={wr:.3f}")
                    if self.serial_enabled and self.serial_output:
                        self.serial_output.send_velocities(wl, wr)

                    # Feature-based bounding box
                    all_x = [l_sh.x * w, r_sh.x * w, l_hip.x * w, r_hip.x * w]
                    all_y = [l_sh.y * h, r_sh.y * h, l_hip.y * h, r_hip.y * h]
                    x_min, x_max = int(min(all_x)), int(max(all_x))
                    y_min, y_max = int(min(all_y)), int(max(all_y))
                    bb_w = x_max - x_min
                    bb_h = y_max - y_min
                    if not self.feature_tracker.initialized:
                        self.feature_tracker.initialize_tracking(frame, (x_min, y_min, bb_w, bb_h))
                    else:
                        self.feature_tracker.initialize_tracking(frame, (x_min, y_min, bb_w, bb_h))

                else:
                    # No pose => fallback or stop motors
                    print("[Tracking] No pose detected; stopping motors.")
                    if self.serial_enabled and self.serial_output:
                        self.serial_output.send_velocities(0, 0)

            # Drawing
            if self.draw_enabled and frame is not None:
                cv2.imshow("Distance & Angle Tracker", frame)
                if self.debug_keys:
                    cv2.waitKey(1)

    # --------------------------------------------------
    #            Cleanup
    # --------------------------------------------------
    def cleanup(self):
        """
        Cleanup all resources: camera, serial, ultrasonic, windows, etc.
        """
        print("Performing final cleanup...")
        self.camera_node.stop()
        if self.serial_output:
            self.serial_output.close()
        self.ultrasonic_sensor.cleanup()
        cv2.destroyAllWindows()
        print("All resources released.")


# ----------------------------------------
#                MAIN
# ----------------------------------------
def main():
    """
    Flow:
      1) Start socket listener thread, passing a reference to the tracker (once created).
      2) Create the DistanceAngleTracker.
      3) By default, we go straight into calibrate_reference() -> start_tracking().
         If you want to wait for actual 'calibrate' or 'start' commands from Bluetooth,
         simply skip calling these in main, or restructure your code:
           - Create the tracker
           - Let the user send 'calibrate' or 'start' from Bluetooth to proceed
    """
    # We'll create the tracker first so we can pass it to the socket thread
    # but we'll only do the main flow if you want an immediate "calibrate -> track".
    args = sys.argv
    if len(args) > 1:
        distance_arg = float(args[1])
    else:
        distance_arg = 0.5

    # Create the main tracking object
    tracker = DistanceAngleTracker(
        camera_index=0,
        target_distance=distance_arg,
        reference_distance=distance_arg,
        polling_interval=0.1,       # how often to compute new control in seconds
        serial_port='/dev/ttyUSB0',
        baudrate=9600,
        serial_enabled=False,       # Change if you have a robot & serial connection
        draw_enabled=True,          # True => show OpenCV windows
        kv=0.1,
        kw=0.00095,
    )

    # Start the socket-listening thread, passing the tracker so commands can set flags
    socket_thread = threading.Thread(target=listen_for_socket_commands, args=(tracker,), daemon=True)
    socket_thread.start()

    try:
        # If you want to automatically calibrate + track at startup,
        # do so here. Otherwise, comment out and rely on "calibrate" or "start" commands.
        tracker.calibrate_reference()

    except KeyboardInterrupt:
        print("\n[Ctrl-C] Caught keyboard interrupt. Exiting gracefully...")

    finally:
        # Always release resources even if code exits unexpectedly
        tracker.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()
