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
    Tracks an object (the calibrated user) using SIFT feature detection.
    Now does full-frame SIFT detection for re-identification, instead of restricting to a bounding box.
    The bounding box from calibration is used to store descriptors, but re-identification scans the entire frame.
    """
    def __init__(self, min_keypoints_threshold=50, ratio_threshold=0.4):
        """
        Initialize the SIFT detector, FLANN matcher, and tracking state.
        :param min_keypoints_threshold: how many good SIFT matches we need to confirm it's the same user.
        :param ratio_threshold: Lowe's ratio threshold for deciding a match is 'good'.
        """
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Descriptors for the user's torso region at calibration.
        self.object_descriptors = None
        # We might store a bounding box if needed, but we won't rely on it for the search.
        self.prev_bbox = None
        self.initialized = False

        # We'll require at least this many matched keypoints to call it "same user".
        self.min_keypoints_threshold = min_keypoints_threshold
        # Lowe's ratio threshold
        self.ratio_threshold = ratio_threshold

    def initialize_object_tracking(self, frame, bbox):
        """
        At calibration time, extract SIFT features only from the bounding box region.
        We'll store those descriptors as the signature for the user.
        """
        self.prev_bbox = bbox
        x, y, w, h = bbox
        user_roi = frame[y:y+h, x:x+w]
        keypoints, descriptors = self.sift.detectAndCompute(user_roi, None)

        if descriptors is None or len(descriptors) == 0:
            return

        self.object_descriptors = descriptors
        self.initialized = True

    def track_object_in_frame(self, frame):
        """
        Attempt to confirm the same user by extracting SIFT from the entire frame,
        matching them against the stored descriptors from calibration.
        If enough matched keypoints, we compute a bounding box of those matched points.
        Returns the bounding box of matched keypoints (x,y,w,h) if recognized, else None.
        """
        if not self.initialized or self.object_descriptors is None:
            return None

        # Extract SIFT from the entire frame
        keypoints, descriptors = self.sift.detectAndCompute(frame, None)
        if descriptors is None or len(descriptors) == 0:
            return None

        try:
            matches = self.flann.knnMatch(descriptors, self.object_descriptors, k=2)
        except:
            return None

        good_points = []
        for i, pair in enumerate(matches):
            if len(pair) == 2:
                d1, d2 = pair[0].distance, pair[1].distance
                if d2 == 0:
                    continue
                ratio = d1 / d2
                if ratio < self.ratio_threshold:
                    # location in the full frame
                    kp_x, kp_y = keypoints[i].pt
                    good_points.append((kp_x, kp_y))

        if len(good_points) < self.min_keypoints_threshold:
            return None

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
        return new_bbox

class CameraManager:
    """
    Manages the camera: initialization, continuous frame capture, and frame retrieval.
    """
    def __init__(self, camera_index=0, width=640, height=360, max_retries=5, delay=1):
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
        cam_process = subprocess.run(["fuser", "/dev/video0"], capture_output=True, text=True)
        if cam_process.stdout.strip():
            os.system("sudo fuser -k /dev/video0")
            time.sleep(1)

    def init_camera(self):
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
        if self.cap is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames_loop, daemon=True)
        self.thread.start()

    def _capture_frames_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
            time.sleep(0.01)

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop_capture(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

class PoseEstimator:
    def __init__(self, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def estimate_pose(self, frame):
        if frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)

class UltrasonicRangeFinder:
    SENSORS = {
        "front": {"TRIG": 11, "ECHO": 12},
        "left":  {"TRIG": 13, "ECHO": 16},
    }

    TIMEOUT = 0.04

    def __init__(self):
        self.sensors = {name: {"TRIG": cfg["TRIG"], "ECHO": cfg["ECHO"]} for name, cfg in UltrasonicRangeFinder.SENSORS.items()}
        self.cached_distances = {name: None for name in self.sensors}
        self.lock = threading.Lock()
        self.chip = lgpio.gpiochip_open(0)
        for name, pins in self.sensors.items():
            lgpio.gpio_claim_output(self.chip, pins["TRIG"])
            lgpio.gpio_claim_input(self.chip, pins["ECHO"])
            lgpio.gpio_write(self.chip, pins["TRIG"], 0)

    def measure_single_distance(self, trig, echo):
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
        dist = self.measure_single_distance(trig, echo)
        with self.lock:
            self.cached_distances[sensor_name] = dist

    def get_all_distances(self):
        threads = []
        for name, pins in self.sensors.items():
            t = threading.Thread(target=self._distance_measure_thread, args=(name, pins["TRIG"], pins["ECHO"]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        return self.cached_distances

    def cleanup_gpio(self):
        lgpio.gpiochip_close(self.chip)

class DistanceEstimator:
    def __init__(self, reference_distance):
        self.reference_distance = reference_distance
        self.reference_height = None

    def calibrate_height(self, vertical_height):
        self.reference_height = vertical_height

    def estimate_distance_from_height(self, current_height):
        if self.reference_height is not None and current_height > 0:
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None

class MotionController:
    def __init__(self,
                 kv,
                 kw,
                 target_distance,
                 distance_tolerance=0.1,
                 angle_tolerance_deg=25):
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.angle_tolerance_deg = angle_tolerance_deg
        self.kv = kv
        self.kw = kw

    def compute_motion_command(self, distance, angle_offset_deg):
        distance_err = 0 if distance is None else distance - self.target_distance
        angle_err = angle_offset_deg
        linear_vel = self.kv * distance_err
        angular_vel = self.kw * angle_err
        return linear_vel, angular_vel

class SerialCommandSender:
    def __init__(self, port='/dev/serial0', baudrate=9600):
        # self.ser = serial.Serial(port, baudrate)
        pass

    def send_wheel_velocities(self, wl, wr):
        msg = f"w_l:{-wr:.3f} w_r:{-wl:.3f}\n"
        # self.ser.write(msg.encode('utf-8'))
        print("[Debug] Serial command:", msg)

    def close_connection(self):
        # if self.ser and self.ser.is_open:
        #     self.ser.close()
        pass

class AutonomousTracker:
    """
    We rely primarily on Mediapipe Pose for tracking.
    SIFT is used solely for re-identification once we are LOST.

    Now, SIFT scans the entire frame instead of a bounding box.
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
                 kw,
                 max_speed):
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()
        self.serial_enabled = serial_enabled
        self.draw_enabled = draw_enabled
        self.debug_keys = True
        self.pending_start = False
        self.pending_stop = False
        self.pending_calibrate = False

        # Tracking states can be "TRACKING" or "LOST".
        self.tracking_state = "LOST"  # default to lost until calibration

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

        # Use a ratio_threshold of 0.7 (more lenient) by default
        self.visual_tracker = VisualFeatureTracker(min_keypoints_threshold=20, ratio_threshold=0.7)

        self.window_name = "Distance & Angle Tracker"

        # We'll store a max_speed from main so we can clamp within the helper.
        self.max_speed = max_speed

    def extract_pose_metrics(self, results, frame):
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]
        l_sh = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        shoulder_y = (l_sh.y + r_sh.y) / 2 * h
        hip_y = (l_hip.y + r_hip.y) / 2 * h
        vert_height = abs(shoulder_y - hip_y)
        user_center_x = (l_sh.x + r_sh.x) / 2 * w
        return vert_height, user_center_x, l_sh, r_sh, l_hip, r_hip, w, h

    def command_socket_listener(self):
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
        self.pending_calibrate = True

    def request_start(self, distance=None):
        if distance is not None:
            self.motion_controller.target_distance = distance
            self.distance_estimator.reference_distance = distance
        self.pending_start = True

    def send_status_update(self, status_message):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect("/tmp/status_socket")
                sock.sendall(status_message.encode())
        except FileNotFoundError:
            pass

    def calibrate_distance_reference(self):
        """
        1) Wait for a valid Pose detection.
        2) Use shoulders & hips to get bounding box.
        3) Initialize the SIFT descriptors for the user (from that bbox), but we'll match against the full frame.
        4) Once done, set self.tracking_state = "TRACKING" (the user is known)
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
                # Extract the vertical height, just like before.
                vert_height, _, l_sh, r_sh, l_hip, r_hip, w, h = self.extract_pose_metrics(results, frame)
                self.distance_estimator.calibrate_height(vert_height)

                # Build bounding box around shoulders and hips.
                all_x = [l_sh.x * w, r_sh.x * w, l_hip.x * w, r_hip.x * w]
                all_y = [l_sh.y * h, r_sh.y * h, l_hip.y * h, r_hip.y * h]
                x_min, x_max = int(min(all_x)), int(max(all_x))
                y_min, y_max = int(min(all_y)), int(max(all_y))
                bb_w = x_max - x_min
                bb_h = y_max - y_min

                self.visual_tracker.initialize_object_tracking(frame, (x_min, y_min, bb_w, bb_h))
                print("[Info] SIFT calibration complete: user descriptors stored.")

                # Once calibrated, we consider ourselves TRACKING.
                self.tracking_state = "TRACKING"
                self.send_status_update("Calibration Completed - Now TRACKING")

                if self.pending_calibrate:
                    self.run_tracking_loop()
                    break

            if self.draw_enabled:
                debug_frame = frame.copy()
                cv2.imshow(self.window_name, debug_frame)
                cv2.waitKey(1)

    def _compute_smoothed_velocities(self, distance_est, user_center_x, w, prev_angle_offset):
        if w == 0:
            return 0, 0, 0, prev_angle_offset

        offset = user_center_x - (w / 2)
        norm_offset = offset / (w / 2)
        angle_deg = max(min(norm_offset * 90, 90), -90)

        if abs(int(angle_deg) - prev_angle_offset) <= 2:
            angle_int = prev_angle_offset
        else:
            angle_int = int(angle_deg)
            prev_angle_offset = angle_int

        linear_vel, angular_vel = self.motion_controller.compute_motion_command(distance_est, angle_int)
        wheelbase_factor = 0.5
        base_speed = linear_vel / wheelbase_factor
        wl = angular_vel + base_speed
        wr = -angular_vel + base_speed

        wl = max(min(wl, self.max_speed), -self.max_speed)
        wr = max(min(wr, self.max_speed), -self.max_speed)

        return wl, wr, angle_int, prev_angle_offset

    def _attempt_pose(self, frame, prev_angle_offset):
        results = self.pose_estimator.estimate_pose(frame)
        if results and results.pose_landmarks:
            vert_height, user_center_x, l_sh, r_sh, l_hip, r_hip, w, h = self.extract_pose_metrics(results, frame)
            distance_est = self.distance_estimator.estimate_distance_from_height(vert_height)

            # Optionally keep a bounding box from pose for debugging
            all_x = [l_sh.x * w, r_sh.x * w, l_hip.x * w, r_hip.x * w]
            all_y = [l_sh.y * h, r_sh.y * h, l_hip.y * h, r_hip.y * h]
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))
            new_bb_w = x_max - x_min
            new_bb_h = y_max - y_min
            if self.visual_tracker.initialized:
                self.visual_tracker.prev_bbox = (x_min, y_min, new_bb_w, new_bb_h)

            wl, wr, angle_int, updated_angle_offset = self._compute_smoothed_velocities(
                distance_est,
                user_center_x,
                w,
                prev_angle_offset
            )
            return wl, wr, distance_est, angle_int, updated_angle_offset, True
        else:
            return 0, 0, None, 0, prev_angle_offset, False

    def run_tracking_loop(self):
        self.send_status_update("Tracking Started")
        last_time_polled = time.time()
        previous_angle_offset = 0

        while True:
            if self.pending_stop:
                self.send_status_update("Tracking Stopped")
                if self.serial_enabled and self.serial_sender:
                    self.serial_sender.send_wheel_velocities(0, 0)
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
                    break

            now = time.time()
            if now - last_time_polled < self.polling_interval:
                continue
            last_time_polled = now

            wl, wr = 0, 0
            distance_est = None
            angle_int = 0

            if self.tracking_state == "LOST":
                if self.visual_tracker.initialized:
                    tracked_bbox = self.visual_tracker.track_object_in_frame(frame)
                    if tracked_bbox is not None:
                        wl, wr, distance_est, angle_int, previous_angle_offset, pose_ok = self._attempt_pose(
                            frame, previous_angle_offset
                        )
                        if pose_ok:
                            self.tracking_state = "TRACKING"
                        else:
                            wl, wr = 0, 0
                    else:
                        wl, wr = 0, 0
                else:
                    wl, wr = 0, 0
            else:
                wl, wr, distance_est, angle_int, previous_angle_offset, pose_ok = self._attempt_pose(
                    frame, previous_angle_offset
                )
                if not pose_ok:
                    self.tracking_state = "LOST"
                    wl, wr = 0, 0

            print(f"[Tracking Info] state={self.tracking_state}, distance={distance_est}, angle={angle_int} deg, wl={wl:.2f}, wr={wr:.2f}")

            if self.serial_enabled and self.serial_sender:
                self.serial_sender.send_wheel_velocities(wl, wr)

            if self.draw_enabled and frame is not None:
                cv2.imshow("Distance & Angle Tracker", frame)
                if self.debug_keys:
                    cv2.waitKey(1)

        # end loop

    def cleanup_resources(self):
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

        if self.serial_sender:
            self.serial_sender.close_connection()
        self.ultrasonic_rangefinder.cleanup_gpio()
        cv2.destroyAllWindows()


def main():
    args = sys.argv
    distance_arg = 0.5
    if len(args) > 1:
        try:
            distance_arg = float(args[1])
        except ValueError:
            pass

    max_speed = 0.4

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
        max_speed=max_speed,
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
