import cv2
import mediapipe as mp
# import serial
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
    Tracks an object within video frames using SIFT feature detection and FLANN matching.
    """
    def __init__(self):
        """
        Initialize the SIFT detector, FLANN matcher, and tracking state.
        """
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.object_descriptors = None
        self.background_descriptors = None
        self.prev_bbox = None
        self.initialized = False

    def initialize_object_tracking(self, frame, bbox):
        """
        Initialize tracking by extracting SIFT features inside (object) and outside (background)
        the provided bounding box.
        Args:
            frame: The input video frame.
            bbox: Tuple (x, y, w, h) representing the object's bounding box.
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

        self.object_descriptors = np.array(obj_desc_list, dtype=np.float32) if obj_desc_list else None
        self.background_descriptors = np.array(bg_desc_list, dtype=np.float32) if bg_desc_list else None

        self.initialized = True

    def track_object_in_frame(self, frame):
        """
        Track the object in a new frame using stored SIFT features.
        Args:
            frame: The current video frame.
        Returns:
            Updated bounding box as (x, y, w, h) if tracking is successful, else None.
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
                d_obj = object_matches[i][0].distance
                if i < len(background_matches) and len(background_matches[i]) > 0:
                    d_bg = background_matches[i][0].distance
                else:
                    d_bg = float('inf')

                if d_bg == 0:
                    continue
                if d_obj / d_bg < 0.5:
                    object_points.append(kp.pt)

        if not object_points:
            return None

        pts = np.array(object_points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        new_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        self.prev_bbox = new_bbox
        return new_bbox


class CameraManager:
    """
    Manages the camera: initialization, continuous frame capture, and frame retrieval.
    """
    def __init__(self, camera_index=0, width=640, height=360, max_retries=5, delay=1):
        """
        Initialize the camera with specified settings.
        """
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

        self.release_camera_resource_if_busy()
        self.init_camera()

    def release_camera_resource_if_busy(self):
        """
        Release the camera if it is in use by another process.
        """
        cam_process = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
        if cam_process.stdout.strip():
            os.system("sudo fuser -k /dev/video0")
            time.sleep(1)

    def init_camera(self):
        """
        Attempt to initialize the camera and set its resolution.
        """
        for attempt in range(1, self.max_retries + 1):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return
            self.release_camera_resource_if_busy()
            time.sleep(self.delay)

        self.cap = None

    def start_capture(self):
        """
        Start capturing frames continuously in a background thread.
        """
        if self.cap is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames_loop, daemon=True)
        self.thread.start()

    def _capture_frames_loop(self):
        """
        Internal loop that continuously captures frames.
        """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def get_latest_frame(self):
        """
        Retrieve the most recent frame captured by the camera.
        Returns:
            The latest frame or None if unavailable.
        """
        with self.lock:
            return self.latest_frame

    def stop_capture(self):
        """
        Stop the frame capture thread and release the camera resource.
        """
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()


class PoseEstimator:
    """
    Estimates human pose from video frames using MediaPipe.
    """
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        """
        Initialize the MediaPipe pose estimator.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def estimate_pose(self, frame):
        """
        Estimate pose landmarks in the given frame.
        Args:
            frame: Input frame in BGR color space.
        Returns:
            Pose estimation results or None if the frame is invalid.
        """
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)


class UltrasonicRangeFinder:
    """
    Measures distances using ultrasonic sensors via GPIO.
    """

    SENSORS = {
        "front": {"TRIG": 11, "ECHO": 12},
        "left":  {"TRIG": 13, "ECHO": 16},
    }

    TIMEOUT = 0.04

    def __init__(self):
        """
        Initialize ultrasonic sensors and claim the necessary GPIO pins.
        """
        self.sensors = {name: {"TRIG": cfg["TRIG"], "ECHO": cfg["ECHO"]} for name, cfg in UltrasonicRangeFinder.SENSORS.items()}
        self.cached_distances = {name: None for name in self.sensors}
        self.lock = threading.Lock()

        self.chip = lgpio.gpiochip_open(0)

        for name, pins in self.sensors.items():
            lgpio.gpio_claim_output(self.chip, pins["TRIG"])
            lgpio.gpio_claim_input(self.chip, pins["ECHO"])
            lgpio.gpio_write(self.chip, pins["TRIG"], 0)

    def measure_single_distance(self, trig, echo):
        """
        Trigger an ultrasonic pulse and measure the distance.
        Args:
            trig: GPIO pin for sending the pulse.
            echo: GPIO pin for receiving the echo.
        Returns:
            Distance in centimeters or None if a timeout occurs.
        """
        lgpio.gpio_write(self.chip, trig, 0)
        time.sleep(0.000002)
        lgpio.gpio_write(self.chip, trig, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(self.chip, trig, 0)
        start_time = time.time()
        while lgpio.gpio_read(self.chip, echo) == 0:
            if time.time() - start_time > UltrasonicRangeFinder.TIMEOUT:
                return None
        pulse_start = time.time()
        while lgpio.gpio_read(self.chip, echo) == 1:
            if time.time() - pulse_start > UltrasonicRangeFinder.TIMEOUT:
                return None
        pulse_end = time.time()
        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2
        return round(distance, 2)

    def _distance_measure_thread(self, sensor_name, trig, echo):
        """
        Helper thread function to measure distance and update the cache.
        """
        dist = self.measure_single_distance(trig, echo)
        with self.lock:
            self.cached_distances[sensor_name] = dist

    def get_all_distances(self):
        """
        Measure distances concurrently for all sensors.
        Returns:
            A dictionary mapping sensor names to their measured distances.
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
        Clean up the GPIO resources by closing the chip.
        """
        lgpio.gpiochip_close(self.chip)


class DistanceEstimator:
    """
    Estimates the physical distance based on a calibrated vertical measurement.
    """
    def __init__(self, reference_distance):
        """
        Initialize with a known reference distance.
        """
        self.reference_distance = reference_distance
        self.reference_height = None

    def calibrate_height(self, vertical_height):
        """
        Calibrate the estimator using a measured vertical height (e.g., shoulder-to-hip).
        """
        self.reference_height = vertical_height

    def estimate_distance_from_height(self, current_height):
        """
        Estimate the distance using the ratio of the calibrated height to the current measured height.
        Args:
            current_height: The current vertical height measurement.
        Returns:
            Estimated distance or None if calibration has not occurred.
        """
        if self.reference_height is not None and current_height > 0:
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None


class MotionController:
    """
    Computes motion commands (linear and angular velocities) based on distance and angle errors.
    """
    def __init__(self,
                 kv,
                 kw,
                 target_distance,
                 distance_tolerance=0.1,
                 angle_tolerance_deg=25):
        """
        Initialize with control gains and target parameters.
        """
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        self.kv = kv
        self.kw = kw

    def compute_motion_command(self, distance, angle_offset_deg):
        """
        Compute linear and angular velocities based on the errors.
        Args:
            distance: Current measured distance.
            angle_offset_deg: Angular offset from center (in degrees).
        Returns:
            A tuple (linear_velocity, angular_velocity).
        """
        distance_err = 0 if distance is None else distance - self.target_distance
        angle_err = angle_offset_deg
        linear_vel = self.kv * distance_err
        angular_vel = self.kw * angle_err
        return linear_vel, angular_vel


class SerialCommandSender:
    """
    Handles serial communication for sending wheel velocity commands.
    """
    def __init__(self, port='/dev/serial0', baudrate=9600):
        """
        Initialize the serial connection.
        """
        self.ser = serial.Serial(port, baudrate)

    def send_wheel_velocities(self, wl, wr):
        """
        Send left and right wheel velocities.
        Args:
            wl: Left wheel velocity.
            wr: Right wheel velocity.
        """
        msg = f"w_l:{-wr:.3f} w_r:{-wl:.3f}\n"
        self.ser.write(msg.encode('utf-8'))

    def close_connection(self):
        """
        Close the serial connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()


class AutonomousTracker:
    """
    Integrates camera input, pose estimation, ultrasonic sensing, and motion control
    to autonomously track an object and command movement.
    """
    def __init__(self,
                 camera_index,
                 target_distance,
                 reference_distance,
                 polling_interval,
                 serial_port,
                 baudrate,
                 serial_enabled,
                 draw_enabled,
                 kv,
                 kw):
        """
        Initialize all sub-modules for autonomous tracking.
        """
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()
        self.serial_enabled = serial_enabled
        self.draw_enabled = draw_enabled
        self.debug_keys = True
        self.pending_start = False
        self.pending_stop = False
        self.pending_calibrate = False

        self.camera_manager = CameraManager(camera_index=camera_index)
        self.camera_manager.start_capture()
        self.pose_estimator = PoseEstimator()
        self.distance_estimator = DistanceEstimator(reference_distance)
        self.motion_controller = MotionController(kv=kv, kw=kw, target_distance=target_distance)
        if self.serial_enabled:
            self.serial_sender = SerialCommandSender(port=serial_port, baudrate=baudrate)
        else:
            self.serial_sender = None
        self.ultrasonic_rangefinder = UltrasonicRangeFinder()
        self.visual_tracker = VisualFeatureTracker()
        self.window_name = "Distance & Angle Tracker"

    def command_socket_listener(self):
        """
        Listen for commands via a UNIX socket and update the tracker accordingly.
        Supported commands:
          - "stop": stop the tracker.
          - "calibrate": begin calibration.
          - "start": start tracking, optionally with a new target distance (e.g., "start:0.7").
        """
        # UNIX domain socket for external command communication.
        COMMAND_SOCKET_PATH = "/tmp/command_socket"

        if os.path.exists(COMMAND_SOCKET_PATH):
            os.remove(COMMAND_SOCKET_PATH)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server_sock:
            server_sock.bind(COMMAND_SOCKET_PATH)
            server_sock.listen(1)
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
        Signal the tracker to stop.
        Wrap closing the MediaPipe Pose in a try-except to avoid errors if it's already closed.
        Also send 0,0 velocities if serial is enabled.
        """
        try:
            if self.pose_estimator and self.pose_estimator.pose:
                self.pose_estimator.pose.close()
                self.pose_estimator.pose = None
        except Exception as e:
            print(f"[Warning] Error closing the MediaPipe Pose object: {e}")

        if self.serial_enabled and self.serial_sender:
            self.serial_sender.send_wheel_velocities(0, 0)
        self.pending_stop = True

    def request_calibration(self):
        """
        Signal the tracker to begin calibration.
        """
        self.pending_calibrate = True

    def request_start(self, distance=None):
        """
        Signal the tracker to start tracking. Optionally update the target distance.
        """
        if distance is not None:
            self.motion_controller.target_distance = distance
            self.distance_estimator.reference_distance = distance
        self.pending_start = True

    def send_status_update(self, status_message):
        """
        Send a status update via a UNIX socket for external monitoring.
        """
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect("/tmp/status_socket")
                sock.sendall(status_message.encode())
        except FileNotFoundError:
            pass

    def calibrate_distance_reference(self):
        """
        Calibrate the distance estimator using pose landmarks.
        """
        self.send_status_update("Calibration Started")
        while True:
            if self.pending_stop:
                self.send_status_update("Calibration Stopped")
                if self.serial_sender:
                    self.serial_sender.send_wheel_velocities(0, 0)
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
                    break
            results = self.pose_estimator.estimate_pose(frame)
            if results and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                shoulder_y = (l_sh.y + r_sh.y) / 2 * h
                hip_y = (l_hip.y + r_hip.y) / 2 * h
                vert_height = abs(shoulder_y - hip_y)
                self.distance_estimator.calibrate_height(vert_height)
                if self.pending_calibrate:
                    self.run_tracking_loop()
                    break
            if self.draw_enabled:
                debug_frame = frame.copy()
                cv2.imshow(self.window_name, debug_frame)
                cv2.waitKey(1)
        

    def run_tracking_loop(self):
        """
        Main loop for tracking that processes frames, estimates pose, computes motion commands,
        and sends wheel velocities.
        """
        self.send_status_update("Tracking Started")
        last_time_polled = time.time()
        previous_angle_offset = 0
        while True:
            # If we have asked to stop, break.
            if self.pending_stop:
                self.send_status_update("Tracking Stopped")
                if self.serial_enabled and self.serial_sender:
                    self.serial_sender.send_wheel_velocities(0, 0)
                break

            # Grab the latest frame.
            frame = self.camera_manager.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # If user pressed 'q', stop.
            if self.debug_keys:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.send_status_update("Tracking Stopped")
                    if self.serial_enabled and self.serial_sender:
                        self.serial_sender.send_wheel_velocities(0, 0)
                    break

            now = time.time()
            # Only process at the polling interval.
            if now - last_time_polled >= self.polling_interval:
                last_time_polled = now
                results = self.pose_estimator.estimate_pose(frame)

                # Defaults:
                wl, wr = 0, 0
                distance_est = None
                angle_int = 0

                if results and results.pose_landmarks:
                    # 1) Compute distance.
                    h, w = frame.shape[:2]
                    landmarks = results.pose_landmarks.landmark
                    l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                    l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                    r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
                    shoulder_y = (l_sh.y + r_sh.y) / 2 * h
                    hip_y = (l_hip.y + r_hip.y) / 2 * h
                    vert_height = abs(shoulder_y - hip_y)
                    distance_est = self.distance_estimator.estimate_distance_from_height(vert_height)

                    # 2) Compute angle offset.
                    user_center_x = (l_sh.x + r_sh.x) / 2 * w
                    offset = user_center_x - (w / 2)
                    norm_offset = offset / (w / 2)
                    angle_deg = max(min(norm_offset * 90, 90), -90)
                    # Smooth angle changes.
                    if abs(int(angle_deg) - previous_angle_offset) <= 2:
                        angle_int = previous_angle_offset
                    else:
                        angle_int = int(angle_deg)
                        previous_angle_offset = angle_int

                    # 3) Calculate velocities.
                    linear_vel, angular_vel = self.motion_controller.compute_motion_command(distance_est, angle_int)
                    wheelbase_factor = 0.5
                    base_speed = linear_vel / wheelbase_factor
                    wl = angular_vel + base_speed
                    wr = -angular_vel + base_speed

                    # 4) Initialize or update visual tracking bounding box.
                    all_x = [l_sh.x * w, r_sh.x * w, l_hip.x * w, r_hip.x * w]
                    all_y = [l_sh.y * h, r_sh.y * h, l_hip.y * h, r_hip.y * h]
                    x_min, x_max = int(min(all_x)), int(max(all_x))
                    y_min, y_max = int(min(all_y)), int(max(all_y))
                    bb_w = x_max - x_min
                    bb_h = y_max - y_min
                    if not self.visual_tracker.initialized:
                        self.visual_tracker.initialize_object_tracking(frame, (x_min, y_min, bb_w, bb_h))
                    else:
                        self.visual_tracker.initialize_object_tracking(frame, (x_min, y_min, bb_w, bb_h))

                # 5) Clamp velocities.
                max_speed = 0.4
                wl = max(min(wl, max_speed), -max_speed)
                wr = max(min(wr, max_speed), -max_speed)

                # 6) Print info.
                print(f"[Tracking Info] distance={distance_est}, angle={angle_int} deg, wl={wl:.2f}, wr={wr:.2f}")

                # 7) Send velocities if serial enabled.
                if self.serial_enabled and self.serial_sender:
                    self.serial_sender.send_wheel_velocities(wl, wr)

            # Draw the frame if desired.
            if self.draw_enabled and frame is not None:
                cv2.imshow("Distance & Angle Tracker", frame)
                if self.debug_keys:
                    cv2.waitKey(1)

    def cleanup_resources(self):
        """
        Clean up resources: stop the camera, close the serial connection, release GPIO, and destroy windows.
        Always send 0,0 velocities before shutting down serial.
        """
        # Always send 0,0 on cleanup if serial is enabled.
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
            except Exception as e:
                print(f"[Warning] Closing pose failed: {e}")

        # Now close the serial.
        if self.serial_sender:
            self.serial_sender.close_connection()
        self.ultrasonic_rangefinder.cleanup_gpio()
        cv2.destroyAllWindows()


def main():
    """
    Main entry point: initialize and start the autonomous tracker.

    We wrap everything in try/except/finally to ensure we always send 0,0 velocities
    and cleanup even if there's a crash or exception.
    """
    args = sys.argv
    distance_arg = 0.5
    if len(args) > 1:
        try:
            distance_arg = float(args[1])
        except ValueError:
            pass

    tracker = AutonomousTracker(
        camera_index=0,
        target_distance=distance_arg,
        reference_distance=distance_arg,
        polling_interval=0.3,
        serial_port='/dev/ttyUSB0',
        baudrate=9600,
        serial_enabled=False,
        draw_enabled=True,
        kv=0.3,
        kw=0.005,
    )

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