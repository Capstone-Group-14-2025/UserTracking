import cv2
import mediapipe as mp
import time
import serial
import RPi.GPIO as GPIO  # For ultrasonic sensors (placeholder)

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


class UltrasonicSensors:
    """
    Ultrasonic Sensor Node (Disabled by default)

    Placeholder class for four ultrasonic sensors at positions:
      - Front Left
      - Front Right
      - Back Left
      - Back Right

    Currently disabled. To enable:
      1. Set up GPIO pins.
      2. Write code to read distance from each sensor.
      3. Use distances in your control logic as needed.
    """

    def __init__(self, enable=False):
        self.enabled = enable

        # Placeholder for the GPIO pins
        self.front_left_pins = (0, 0)   # (TRIG, ECHO)
        self.front_right_pins = (0, 0)
        self.back_left_pins = (0, 0)
        self.back_right_pins = (0, 0)

        if self.enabled:
            # Initialize GPIO
            GPIO.setmode(GPIO.BCM)
            # Example placeholder code; actual pins depend on your wiring:
            # GPIO.setup(self.front_left_pins[0], GPIO.OUT)
            # GPIO.setup(self.front_left_pins[1], GPIO.IN)
            pass

    def read_sensors(self):
        """
        Returns a dictionary of distances measured by the ultrasonic sensors.
        Currently returns dummy values (None or 0) as this is disabled.
        """
        if not self.enabled:
            return {
                "front_left": None,
                "front_right": None,
                "back_left": None,
                "back_right": None
            }

        # Actual reading code would go here
        # e.g.:
        # front_left_dist = self._read_single_ultrasonic(self.front_left_pins)
        # ...
        return {
            "front_left": 0,
            "front_right": 0,
            "back_left": 0,
            "back_right": 0
        }

    def _read_single_ultrasonic(self, pins):
        """
        Example private method to read a single ultrasonic sensor.
        Not implemented in this placeholder. 
        """
        pass

    def cleanup(self):
        """
        Cleans up GPIO if enabled.
        """
        if self.enabled:
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
                 target_distance=0.5,
                 distance_tolerance=0.1,
                 angle_tolerance_deg=25,
                 kv=1.0,
                 kw=1.0):
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
		return
        self.ser = serial.Serial(port, baudrate)

    def send_velocities(self, linear_vel, angular_vel):
        """
        Sends the linear and angular velocities over serial as a
        comma-separated string (e.g. "0.10,3.50").
        """
		print(linear_vel)
		print(angular_vel)
		return
		
        msg = f"{linear_vel:.3f},{angular_vel:.3f}\n"
        self.ser.write(msg.encode('utf-8'))

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
                 baudrate=9600):
        # Interval for processing
        self.polling_interval = polling_interval
        self.last_poll_time = time.time()

        # Initialize nodes
        self.camera_node = CameraNode(camera_index=camera_index)
        self.pose_node = PoseEstimationNode()
        self.distance_calculator = DistanceCalculator(reference_distance=reference_distance)
        self.movement_controller = MovementController(target_distance=target_distance)
        self.serial_output = SerialOutput(port=port, baudrate=baudrate)
        self.ultrasonic_node = UltrasonicSensors(enable=False)  # Disabled for now

        # For visualization / debugging
        self.window_name = "Distance & Angle Tracker"

    def calibrate_reference(self):
        """
        Allows the user to calibrate the reference height by pressing 'c'.
        Press 'q' to exit early.
        """
        print("Stand at the known distance and press 'c' to calibrate reference height (or 'q' to quit).")

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

                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.distance_calculator.calibrate(vertical_height)
                    break
                if key == ord('q'):
                    break
            else:
                cv2.imshow("Calibrate Reference Distance", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyWindow("Calibrate Reference Distance")

    def start_tracking(self):
        """
        Main loop that captures frames, estimates distance & angle,
        computes velocities, and sends them over serial.
        Press 'q' to quit.
        """
        while True:
            frame = self.camera_node.get_frame()
            if frame is None:
                print("[ERROR] Could not read frame.")
                break

            # Check user input at any time
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate_reference()

            current_time = time.time()
            if (current_time - self.last_poll_time) >= self.polling_interval:
                self.last_poll_time = current_time

                frame_height, frame_width = frame.shape[:2]
                results = self.pose_node.process_frame(frame)

                # Read ultrasonic data (unused in control for now)
                sensor_data = self.ultrasonic_node.read_sensors()
                # print(sensor_data)  # For debugging if needed

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
                    linear_vel, angular_vel = self.movement_controller.compute_control(distance, angle_offset_deg)

                    # Send to serial
                    self.serial_output.send_velocities(linear_vel, angular_vel)

                    # OPTIONAL: Visual feedback
                    self._draw_info(frame, distance, linear_vel, angular_vel, angle_offset_deg, user_center_x, shoulder_y)
                else:
                    # If pose not detected
                    cv2.putText(frame, "No Pose Detected", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(self.window_name, frame)

        # Cleanup
        self.camera_node.release()
        self.serial_output.close()
        self.ultrasonic_node.cleanup()
        cv2.destroyAllWindows()

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
    # Example usage
    tracker = DistanceAngleTracker(
        camera_index=0,
        target_distance=0.5,        # Desired distance to maintain
        reference_distance=0.5,     # Known distance for calibration
        polling_interval=0.1,       # Interval between processing frames
        port='/dev/tty.usbserial-022680BC',  # Update with your actual port
        baudrate=9600
    )

    # First, calibrate reference height
    tracker.calibrate_reference()

    # Start the main tracking loop
    tracker.start_tracking()
