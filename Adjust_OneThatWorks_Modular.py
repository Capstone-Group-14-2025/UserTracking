import cv2
import mediapipe as mp
import serial
import sys
import RPi.GPIO as GPIO
import time
import statistics
import socket

# Constants for GPIO pins
SENSORS = {
    "front": {"TRIG": 11, "ECHO": 12},
    "left": {"TRIG": 13, "ECHO": 16},
    "right": {"TRIG": 17, "ECHO": 18},
    "back": {"TRIG": 19, "ECHO": 20},
}
BUFFER_SIZE = 5
MEASUREMENTS = 5
TIMEOUT = 0.04  # 40ms timeout

# GPIO setup
GPIO.setmode(GPIO.BCM)
for sensor in SENSORS.values():
    GPIO.setup(sensor["TRIG"], GPIO.OUT)
    GPIO.setup(sensor["ECHO"], GPIO.IN)

class CameraNode:
    """
    Camera Input Node

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

    def measure_distance(self, trig_pin, echo_pin):
        GPIO.output(trig_pin, True)
        time.sleep(0.00001)  # 10 microseconds
        GPIO.output(trig_pin, False)

        start_time = time.time()
        timeout_time = start_time + TIMEOUT

        while GPIO.input(echo_pin) == 0:
            if time.time() > timeout_time:
                return None
            start_time = time.time()

        while GPIO.input(echo_pin) == 1:
            if time.time() > timeout_time:
                return None
            stop_time = time.time()

        time_elapsed = stop_time - start_time
        distance = (time_elapsed * 34300) / 2

        return distance if 2 <= distance <= 500 else None

    def read_distances(self):
        distances = {}
        for name, pins in self.sensors.items():
            current_distances = []
            for _ in range(MEASUREMENTS):
                dist = self.measure_distance(pins["TRIG"], pins["ECHO"])
                if dist is not None:
                    current_distances.append(dist)
                time.sleep(0.05)

            if current_distances:
                median_dist = statistics.median(current_distances)
                self.distance_buffers[name].append(median_dist)
                if len(self.distance_buffers[name]) > BUFFER_SIZE:
                    self.distance_buffers[name].pop(0)
                distances[name] = sum(self.distance_buffers[name]) / len(self.distance_buffers[name])
            else:
                distances[name] = None

        return distances

    def cleanup(self):
        GPIO.cleanup()

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
                 polling_interval=0.1,
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

    def start_tracking(self, draw_enabled = False):
        """
        Main loop that captures frames, estimates distance & angle,
        computes velocities, and sends them over serial.
        Press 'q' to quit.
        """
        sentStatus = False
        while True:
            frame = self.camera_node.get_frame()
            if frame is None:
                print("[ERROR] Could not read frame.")
                break

            distances = self.ultrasonic_sensor.read_distances()
            print("Ultrasonic distances:", distances)

            # Check user input at any time
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.signal_status("Tracking Stopped")
                if(self.serial_enabled):
                    self.serial_output.send_velocities(0, 0)
                break
            elif key == ord('c'):
                self.signal_status("Calibration")
                self.calibrate_reference()

            current_time = time.time()
            if (current_time - self.last_poll_time) >= self.polling_interval:
                self.last_poll_time = current_time

                frame_height, frame_width = frame.shape[:2]
                results = self.pose_node.process_frame(frame)

                if not sentStatus:
                    self.signal_status("Tracking Started")
                    sentStatus = True

                if results.pose_landmarks:
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



                    # Calculate velocities
                    angle_offset_int = int(angle_offset_deg)
                    if -10 <= angle_offset_int <= 10:
                        angle_offset_int = 0
                    linear_vel, angular_vel = self.movement_controller.compute_control(distance, angle_offset_int)
                    radius = 0.5
                    w_hat_l = linear_vel/radius
                    w_hat_r = w_hat_l

                    wl = angular_vel + w_hat_l
                    wr = -angular_vel + w_hat_r
                    #wr = wr * (3.141592653589793 / 180)  # Convert degrees to radians
                    wr = min(wr, 0.5)
                    wl = min(wl, 0.5)
                    wr = max(wr, -0.5)
                    wl = max(wl, -0.5)
                    print("Angular Velocity:",angular_vel)
                    print("Angle Offset:",angle_offset_int)
                    print("wr:",wr , " " ,"wl:",wl)

                    # Send to serial
                    if(self.serial_enabled):
                        self.serial_output.send_velocities(wl, wr)

                    # OPTIONAL: Visual feedback
                    if(self.draw_enabled):
                        self._draw_info(frame, distance, linear_vel, angular_vel, angle_offset_deg, user_center_x, shoulder_y)
                else:
                    # If pose not detected
                    cv2.putText(frame, "No Pose Detected", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(self.window_name, frame)

        # Cleanup
        self.camera_node.release()
        if(self.serial_enabled):
            self.serial_output.close()
        self.ultrasonic_sensor.cleanup()
        #cv2.destroyAllWindows()

    def _draw_info(self, frame, distance, linear_vel, angular_vel, angle_offset_deg, user_center_x, user_center_y):
        """
        Helper method for drawing text overlays on the frame to
        display debugging info: distance, velocities, offsets, etc.
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
        cv2.putText(frame, f"User Pos: ({int(user_center_x)}, {int(user_center_y)})",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)


# ----------------------------
#             MAIN
# ----------------------------
if __name__ == "__main__":
    args = sys.argv
    distance = float(args[1])

    #check distance
    checkDistance = distance > 0 and distance < 2
    if not checkDistance:
        distance = 0.5
    
    tracker = DistanceAngleTracker(
        camera_index=0,
        target_distance=0.5,        # Desired distance to maintain
        reference_distance=distance,     # Known distance for calibration
        polling_interval=0.1,       # Interval between processing frames
        port='/dev/ttyUSB0',  # Update with your actual port
        baudrate=9600,
        serial_enabled=False,
        draw_enabled=False,
        kv=0.8,
        kw=0.005,
    )

    # First, calibrate reference height
    tracker.calibrate_reference()

    print("Ending Script")
