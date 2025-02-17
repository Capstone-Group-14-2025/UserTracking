import cv2
import mediapipe as mp
import serial
import sys
import RPi.GPIO as GPIO
import time
import statistics
import socket
import numpy as np
import threading
import fcntl
import os
import subprocess

# Constants for GPIO pins
SENSORS = {
    "front": {"TRIG": 11, "ECHO": 12},
    "left": {"TRIG": 13, "ECHO": 16},
    # "right": {"TRIG": 17, "ECHO": 18},
    # "back": {"TRIG": 19, "ECHO": 20},
}
BUFFER_SIZE = 5
MEASUREMENTS = 5
TIMEOUT = 0.04  # 40ms timeout

# GPIO setup

import lgpio

# Open GPIO chip (Pi 5 uses /dev/gpiochip0)
h = lgpio.gpiochip_open(0)

# Define GPIO pins
TRIG = 11  # Example pin number
ECHO = 12  # Example pin number

# Set TRIG as output and ECHO as input
lgpio.gpio_claim_output(h, TRIG)
lgpio.gpio_claim_input(h, ECHO)

# ---------------------------------------------------
#             SIFT-BASED FEATURE TRACKER
# ---------------------------------------------------
class FeatureBasedTracker:
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
        # Detect keypoints in entire frame
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

        # KNN match to object and background
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

        # Use min/max to define bounding box
        points = np.array(object_points)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        new_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        self.prev_bbox = new_bbox
        return new_bbox


class CameraNode:
    """
    Camera Input Node for a USB Webcam using OpenCV.

    - Detects if the camera is already in use and forces release.
    - Retries multiple times to ensure the camera is freed before use.
    """

    def __init__(self, camera_index=0, width=320, height=240, max_retries=5, delay=1):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.max_retries = max_retries
        self.delay = delay
        self.cap = None

        print("🔄 Checking if camera is in use...")
        self.release_camera_if_busy()

        # Try initializing the webcam with retries
        self.initialize_camera()

    def release_camera_if_busy(self):
        """Force release the camera if it's locked by another process."""
        cam_process = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
        if cam_process.stdout.strip():
            print(f"⚠️ Camera is in use by process {cam_process.stdout.strip()}. Releasing...")
            os.system("sudo fuser -k /dev/video0")  # Kill process using camera
            time.sleep(1)  # Give the system time to release it
        else:
            print("✅ Camera is free.")

    def initialize_camera(self):
        """Tries to initialize the webcam, retrying if necessary."""
        for attempt in range(1, self.max_retries + 1):
            print(f"📷 Initializing Webcam (Attempt {attempt}/{self.max_retries})...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)  # Use V4L2 explicitly

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print("✅ Webcam initialized successfully.")
                return  # Success, exit function

            print(f"❌ Failed to open webcam on attempt {attempt}. Retrying...")
            self.release_camera_if_busy()
            time.sleep(self.delay)

        print("❌ Error: Could not open webcam after multiple attempts.")
        self.cap = None  # Mark as unavailable

    def get_frame(self):
        """Returns a frame from the webcam (BGR format)."""
        if self.cap is None:
            print("❌ Error: Camera not initialized.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("❌ Error: Could not read frame.")
            return None

        return frame

    def release(self):
        """Releases the camera resource."""
        if self.cap:
            self.cap.release()
            print("🔄 Camera released.")
        else:
            print("⚠️ Camera was not initialized.")

class PoseEstimationNode:
    """
    Pose Estimation Node
    
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


class UltrasonicSensor:
    def __init__(self):
        self.sensors = {name: {"TRIG": config["TRIG"], "ECHO": config["ECHO"]} for name, config in SENSORS.items()}
        self.distance_buffers = {name: [] for name in SENSORS}
        self.cached_distances = {name: None for name in SENSORS}
        self.lock = threading.Lock()

    def measure_distance(self, trig_pin, echo_pin):
        return 0
        """
        Measure distance using an ultrasonic sensor with `lgpio`.
        """
        lgpio.gpio_write(h, trig_pin, 1)
        time.sleep(0.00001)  # 10 microseconds pulse
        lgpio.gpio_write(h, trig_pin, 0)

        start_time = time.time()
        timeout_time = start_time + TIMEOUT

        while lgpio.gpio_read(h, echo_pin) == 0:
            if time.time() > timeout_time:
                return None
            start_time = time.time()

        while lgpio.gpio_read(h, echo_pin) == 1:
            if time.time() > timeout_time:
                return None
            stop_time = time.time()

        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2  # Convert to cm

        return distance if 2 <= distance <= 500 else None  # Valid range

    def async_measure(self, name, trig_pin, echo_pin):
        """
        Asynchronous measurement thread for each sensor.
        """
        dist = self.measure_distance(trig_pin, echo_pin)
        if dist is not None:
            with self.lock:
                self.cached_distances[name] = dist

    def read_distances(self):
        """
        Start multiple threads to measure distances from all sensors.
        """
        threads = []
        for name, pins in self.sensors.items():
            thread = threading.Thread(target=self.async_measure, args=(name, pins['TRIG'], pins['ECHO']))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return self.cached_distances

    def cleanup(self):
        """
        Release the GPIO chip on cleanup.
        """
        lgpio.gpiochip_close(h)


class DistanceCalculator:
    """
    Distance Calculator Node
    
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
        Returns an estimated distance in meters based on the ratio of 
        current_height to the reference_height.
        """
        if self.reference_height is not None and current_height > 0:
            # d_estimated = d_ref * (h_ref / h_current)
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None


class MovementController:
    """
    Movement Controller Node
    
    Calculates linear and angular velocities given distance and angle errors.
    Optionally, can check ultrasonic data for additional collision avoidance
    or advanced logic in the future.
    """

    def __init__(self,
                 kv=1.0,
                 kw=1.0,
                 target_distance=0.5,
                 distance_tolerance=0.1,
                 angle_tolerance_deg=25
                 ):
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        self.kv = kv  # Gain for linear velocity
        self.kw = kw  # Gain for angular velocity

    def compute_control(self, distance, angle_offset_deg):
        """
        Given the current distance (meters) and angle offset (degrees),
        returns the linear and angular velocities to send to the robot.
        """
        # Distance error
        if distance is None:
            distance_error = 0
        else:
            distance_error = distance - self.target_distance

        # Angle error in degrees
        angle_error_deg = angle_offset_deg

        # Convert degrees to radians for internal calculations if desired
        # or simply keep it in degrees if your system expects that.
        # For demonstration, we'll keep degrees but still do the scaling.
        
        # Linear velocity
        linear_vel = self.kv * distance_error
        # Angular velocity
        angular_vel = self.kw * angle_error_deg

        return linear_vel, angular_vel


class SerialOutput:
    """
    Serial Output Node
    
    Sends commands (in this case, linear and angular velocities) to
    the robot over a serial connection.
    """

    def __init__(self, port, baudrate=9600):
	
        self.ser = serial.Serial(port, baudrate)
        return

    def send_velocities(self, wl, wr):
        """
        Sends the linear and angular velocities over serial as a
        formatted string (e.g., "w_l:0.1 w_r:3.5").
        """
        msg = f"w_l:{wl:.1f} w_r:{wr:.1f}\n"

        self.ser.write(msg.encode('utf-8'))
        #print("wl:",wl)
        #print("wr:",wr)
        return

    def close(self):
        """
        Closes the serial connection.
        """
        self.ser.close()


class DistanceAngleTracker:
    """
    Main Node that orchestrates everything:
      1. Capture frames
      2. Detect pose (shoulder & hip positions)
      3. Estimate distance
      4. Compute linear & angular velocities
      5. Send to serial output
      6. (Optionally) read from ultrasonic sensors
    """

    def __init__(self,
                 camera_index=0,
                 target_distance=0.5,
                 reference_distance=0.5,
                 polling_interval=0.3,
                 port='/dev/tty.usbserial-022680BC',
                 baudrate=9600,
                 serial_enabled = False,
                 draw_enabled = False,
                 kv=1.0,
                 kw=1.0):
        # Interval for processing
        self.draw_enabled = draw_enabled
        self.serial_enabled = serial_enabled
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()

        self.last_distance = None
        self.last_linear_vel = None
        self.last_angular_vel = None
        self.last_angle_offset_deg = None
        self.last_user_center_x = None
        self.last_user_center_y = None
        self.last_bbox = None
        self.last_pose_detected = False


        # Initialize nodes
        print("Initializing Camera")
        self.camera_node = CameraNode(camera_index=camera_index)
        print("Initializing Pose Node")

        self.pose_node = PoseEstimationNode()
        print("Initializing Distance Calculator")
        self.distance_calculator = DistanceCalculator(reference_distance=reference_distance)
        print("Initializing Controller")
        self.movement_controller = MovementController(kv=kv,kw=kw,target_distance=target_distance)
        if(serial_enabled):
            print("Initializing Serial Node")
            self.serial_output = SerialOutput(port=port, baudrate=baudrate)
        print("Initializing Ultrasonic Node")
        self.ultrasonic_sensor = UltrasonicSensor()
        print("Initializing Feature-Based Tracker")
        self.feature_tracker = FeatureBasedTracker()
        self.sift_bbox = None  # We'll store the current bounding box from SIFT
        # For visualization / debugging
        self.window_name = "Distance & Angle Tracker"
    
    def signal_status(self, status):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            try:
                sock.connect("/tmp/status_socket")
                sock.sendall(status.encode())
            except FileNotFoundError:
                print("Socket not available")

    def calibrate_reference(self):
        """
        Allows the user to calibrate the reference height by pressing 'c'.
        Press 'q' to exit early.
        """
        print("Stand at the known distance and press 'c' to calibrate reference height (or 'q' to quit).")
        sentStatus = False
        
        while True:
            frame = self.camera_node.get_frame()
            if frame is None:
                print("Could not access the camera.")
                break

            results = self.pose_node.process_frame(frame)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Shoulder points
                left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                # Hip points
                left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                # Calculate the vertical height in pixels
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0]
                hip_y = (left_hip.y + right_hip.y) / 2 * frame.shape[0]
                vertical_height = abs(shoulder_y - hip_y)

                # Draw calibration line
                shoulder_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
                hip_center_x = int((left_hip.x + right_hip.x) / 2 * frame.shape[1])
                cv2.line(frame,
                         (shoulder_center_x, int(shoulder_y)),
                         (hip_center_x, int(hip_y)),
                         (0, 255, 0), 2)
                cv2.putText(frame, "Calibrate Height: Press 'c' to set", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Calibrate Reference Distance", frame)
                if not sentStatus:
                    self.signal_status("Calibration Started")
                    sentStatus = True
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.distance_calculator.calibrate(vertical_height)
                    self.start_tracking()
                    break
                if key == ord('q'):
                    self.signal_status("Calibration Stopped")
                    break
            else:
                cv2.imshow("Calibrate Reference Distance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyWindow("Calibrate Reference Distance")

    def start_tracking(self, draw_enabled=False):
        """
        Main loop that captures frames, estimates distance & angle,
        computes velocities, and sends them over serial.
        Press 'q' to quit.
        """
        sentStatus = False
        skip_polling_check = False
        previous_angle_offset_int = 0

        # Last known values to maintain persistent drawing
        last_distance = 0
        last_linear_vel = 0
        last_angular_vel = 0
        last_angle_offset_deg = 0
        last_user_center_x = 0
        last_user_center_y = 0
        last_pose_detected = False
        last_bbox = None

        while True:
            frame = self.camera_node.get_frame()
            if frame is None:
                print("[ERROR] Could not read frame.")
                break

            distances = self.ultrasonic_sensor.read_distances()
            #print("Ultrasonic distances:", distances)  # CM is measurement
            #if any(distance is not None and distance < 30 for distance in distances.values() and False):
            #    print("YOU ARE TOO CLOSE")
            #    if self.serial_enabled:
            #        self.serial_output.send_velocities(0, 0)

            # Check user input at any time
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                
                if self.serial_enabled:
                    self.serial_output.send_velocities(0, 0)
                self.signal_status("Tracking Stopped")
                skip_polling_check = True
                
                break
            elif key == ord('c'):
                self.signal_status("Calibration")
                self.calibrate_reference()

            current_time = time.time()
            should_update = not skip_polling_check and (current_time - self.last_poll_time) >= self.polling_interval

            if should_update:
                self.last_poll_time = current_time

                frame_height, frame_width = frame.shape[:2]
                results = self.pose_node.process_frame(frame)

                # Default values if no Pose is detected
                distance = 0
                linear_vel = 0
                angular_vel = 0
                angle_offset_deg = 0
                user_center_x = frame_width / 2
                user_center_y = frame_height / 2
                pose_detected = False
                bbox = None

                if not sentStatus:
                    self.signal_status("Tracking Started")
                    sentStatus = True

                if results.pose_landmarks:
                    pose_detected = True
                    landmarks = results.pose_landmarks.landmark
                    # Shoulders
                    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    # Hips
                    left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

                    # Compute the vertical height
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
                    hip_y = (left_hip.y + right_hip.y) / 2 * frame_height
                    vertical_height = abs(shoulder_y - hip_y)

                    # Estimate user distance
                    distance = self.distance_calculator.estimate_distance(vertical_height)

                    # Compute horizontal offset & angle offset
                    user_center_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_width
                    horizontal_offset = user_center_x - (frame_width / 2)
                    # Normalize offset in range [-1, 1]
                    normalized_offset = horizontal_offset / (frame_width / 2)
                    # Convert to degrees (approx. ±90 degrees)
                    angle_offset_deg = max(min(normalized_offset * 90, 90), -90)

                    print("actual angle:", angle_offset_deg)

                    # Calculate velocities
                    angle_offset_int = int(angle_offset_deg)
                    if abs(angle_offset_int - previous_angle_offset_int) <= 2.5:
                        angle_offset_int = previous_angle_offset_int
                    else:
                        previous_angle_offset_int = angle_offset_int

                    linear_vel, angular_vel = self.movement_controller.compute_control(distance, angle_offset_int)
                    radius = 0.5
                    w_hat_l = linear_vel / radius
                    w_hat_r = w_hat_l

                    wl = angular_vel + w_hat_l
                    wr = -angular_vel + w_hat_r
                    wr = min(wr, 0.08)
                    wl = min(wl, 0.08)
                    wr = max(wr, -0.08)
                    wl = max(wl, -0.08)
                    print("Angular Velocity:", angular_vel)
                    print("Angle Offset:", angle_offset_int)
                    print("wr:", wr, " ", "wl:", wl)

                    # Send to serial
                    if self.serial_enabled:
                        self.serial_output.send_velocities(wl, wr)

                    # --- SIFT bounding box update ---
                    # Define a bounding box around shoulders & hips
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

                    # Initialize or re-initialize tracker
                    if not self.feature_tracker.initialized:
                        self.feature_tracker.initialize_tracking(frame, (x_min, y_min, w, h))
                    else:
                        self.feature_tracker.initialize_tracking(frame, (x_min, y_min, w, h))

                    bbox = (x_min, y_min, w, h)

                else:
                    # --- If Mediapipe fails, fallback to feature-based tracking ---
                    #new_bbox = self.feature_tracker.track_object(frame)
                    #if new_bbox is None:
                    #     bbox = new_bbox
                    # else:
                    cv2.putText(frame, "Tracking lost!", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    if self.serial_enabled:
                        self.serial_output.send_velocities(0, 0)

                # Store the last known values for frames when polling skips updates
                last_distance = distance
                last_linear_vel = linear_vel
                last_angular_vel = angular_vel
                last_angle_offset_deg = angle_offset_deg
                last_user_center_x = user_center_x
                last_user_center_y = user_center_y
                last_pose_detected = pose_detected
                last_bbox = bbox

            # --- Always draw overlays using the last known values ---
            if self.draw_enabled:
                self._draw_info(
                    frame=frame,
                    distance=last_distance,
                    linear_vel=last_linear_vel,
                    angular_vel=last_angular_vel,
                    angle_offset_deg=last_angle_offset_deg,
                    user_center_x=last_user_center_x,
                    user_center_y=last_user_center_y,
                    bbox=last_bbox,
                    pose_detected=last_pose_detected
                )

                cv2.imshow(self.window_name, frame)

        # Cleanup
        self.camera_node.release()
        if self.serial_enabled:
            self.serial_output.close()
        self.ultrasonic_sensor.cleanup()


    def _draw_info(self, frame, distance, linear_vel, angular_vel, angle_offset_deg, user_center_x, user_center_y, bbox=None, pose_detected=False):
        """
        Draws:
          - textual info (distance, velocities, etc.)
          - bounding box from SIFT, if available
          - a "torso line" if pose_detected is True
        Ensures persistent overlay when polling skips frames.
        """

        # If no update (polling skipped), reuse last known values
        if distance is None:
            distance = self.last_distance if self.last_distance is not None else 0
        else:
            self.last_distance = distance

        if linear_vel is None:
            linear_vel = self.last_linear_vel if self.last_linear_vel is not None else 0
        else:
            self.last_linear_vel = linear_vel

        if angular_vel is None:
            angular_vel = self.last_angular_vel if self.last_angular_vel is not None else 0
        else:
            self.last_angular_vel = angular_vel

        if angle_offset_deg is None:
            angle_offset_deg = self.last_angle_offset_deg if self.last_angle_offset_deg is not None else 0
        else:
            self.last_angle_offset_deg = angle_offset_deg

        if user_center_x is None:
            user_center_x = self.last_user_center_x if self.last_user_center_x is not None else frame.shape[1] // 2
        else:
            self.last_user_center_x = user_center_x

        if user_center_y is None:
            user_center_y = self.last_user_center_y if self.last_user_center_y is not None else frame.shape[0] // 2
        else:
            self.last_user_center_y = user_center_y

        if bbox is None:
            bbox = self.last_bbox if self.last_bbox is not None else None
        else:
            self.last_bbox = bbox

        if not pose_detected:
            pose_detected = self.last_pose_detected
        else:
            self.last_pose_detected = pose_detected

        text_color = (0, 255, 255)
        font_scale = 0.5

        # Draw text overlays
        cv2.putText(frame, f"Distance: {distance:.2f} m",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Linear Vel: {linear_vel:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Angular Vel: {angular_vel:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"Angle Offset: {angle_offset_deg:.2f} deg",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
        cv2.putText(frame, f"User Ctr: ({int(user_center_x)}, {int(user_center_y)})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        # Draw bounding box if available
        if bbox is not None:
            (bx, by, bw, bh) = bbox
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        # Draw a small line to show "torso"
        if pose_detected:
            cx = int(user_center_x)
            cy = int(user_center_y)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 2)


# ----------------------------
#             MAIN
# ----------------------------
if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        distance_arg = float(args[1])
    else:
        distance_arg = 0.5

    print(distance_arg)

    tracker = DistanceAngleTracker(
        camera_index=0,
        target_distance=distance_arg,         # Desired distance to maintain
        reference_distance=distance_arg,  # Known distance for calibration
        polling_interval=0,
        port='/dev/ttyUSB0',
        baudrate=9600,
        serial_enabled=True,
        draw_enabled=False,
        kv=0.1,
        kw=0.00095,

    )

    # First, calibrate reference heightq
    tracker.calibrate_reference()

    print("Ending Script")
