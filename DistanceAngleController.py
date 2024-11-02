import cv2
import mediapipe as mp
import time

class DistanceAngleController:

    """
    Distance Angle Controller (User Tracker)

    This program captures video input to track a user in real-time, estimate their distance, and calculate
    angular offsets. It uses a reference distance for calibration, enabling the robot to adjust its position 
    to maintain a target distance and alignment with the user. 

    Features:
    - Captures and processes video frames at a specified interval to reduce computation load.
    - Utilizes MediaPipe's pose detection for user tracking, measuring shoulder-to-hip height for distance estimation.
    - Controls movement output, either as PWM signals for a the Jaguar v2 chassis controller or descriptive movement instructions.
    - Adjustable parameters include the target distance, angle tolerance, and polling interval.

    Usage:
    - Set a reference distance for calibration.
    - Choose the type of output ('pwm' for control signals or 'movement' for descriptive instructions).
    - The robot adjusts its position based on distance and angle, tracking the user while maintaining a desired spacing.
    """

    def __init__(self):
        # set up ideal distance and tolerance for tracking
        self.target_distance = 0.8      # distance to maintain from user in meters
        self.reference_distance = 0.5   # reference distance for initial calibration
        self.reference_height = None    # stores shoulder-to-hip height at the reference distance
        self.distance_tolerance = 0.1   # acceptable deviation from the target distance
        self.angle_tolerance = 5        # acceptable angle deviation in degrees

        # set how often commands should be processed (e.g., 4 times per second = 0.25)
        self.polling_interval = 0.25  # interval in seconds
        self.last_poll_time = time.time()

        # set up the camera for lower resolution to speed up processing
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        # initialize pose detection with low thresholds for faster, lightweight tracking
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    def set_reference_distance(self):
        print("Stand at a known distance and press 'c' to calibrate reference height.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Could not access the camera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # calculate shoulder-to-hip height for calibration
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame.shape[0]
                hip_y = (left_hip.y + right_hip.y) / 2 * frame.shape[0]
                vertical_height = abs(shoulder_y - hip_y)

                # visualize calibration line on frame
                cv2.line(frame, 
                         (int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]), int(shoulder_y)),
                         (int((left_hip.x + right_hip.x) / 2 * frame.shape[1]), int(hip_y)),
                         (0, 255, 0), 2)
                cv2.putText(frame, "Calibrate Height", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Set Reference Distance", frame)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    self.reference_height = vertical_height
                    print(f"Reference height set at {self.reference_height} pixels for {self.reference_distance} meters.")
                    break
            
            # exit on q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Set Reference Distance")

    def estimate_distance(self, current_height):
        # estimate distance based on calibrated height
        if self.reference_height is not None and current_height > 0:
            return round(self.reference_distance * (self.reference_height / current_height), 2)
        return None

    def control_movement(self, distance, horizontal_offset, frame_width, print_type="movement"):
        # determine how far off the target distance and angle are
        movement_distance = distance - self.target_distance if distance is not None else 0
        normalized_offset = horizontal_offset / (frame_width / 2)
        angle_offset = max(min(normalized_offset * 90, 90), -90)

        # calculate PWM values based on movement needs ### THIS NEEDS REVIEW FROM ALI AND YEAB
        pwm_forward = min(abs(int(movement_distance * 255 / 1.0)), 255)
        pwm_turn = min(abs(int(angle_offset * 255 / 90)), 255)

        # print PWM or movement information based on preference
        if print_type == "pwm":
            # display PWM commands
            if abs(angle_offset) > self.angle_tolerance:
                if angle_offset > 0:
                    print(f"PWM: Rotate Right - Left Track: {pwm_turn}, Right Track: {-pwm_turn}")
                else:
                    print(f"PWM: Rotate Left - Left Track: {-pwm_turn}, Right Track: {pwm_turn}")
            elif abs(movement_distance) > self.distance_tolerance:
                if movement_distance > 0:
                    print(f"PWM: Move Forward - Left Track: {pwm_forward}, Right Track: {pwm_forward}")
                else:
                    print(f"PWM: Move Backward - Left Track: {-pwm_forward}, Right Track: {-pwm_forward}")
            else:
                print("PWM: Stop")

        elif print_type == "movement":
            # display high-level movement description
            if abs(angle_offset) > self.angle_tolerance:
                if angle_offset > 0:
                    print(f"Movement: Rotate Right by {angle_offset:.2f} degrees")
                else:
                    print(f"Movement: Rotate Left by {abs(angle_offset):.2f} degrees")
            elif abs(movement_distance) > self.distance_tolerance:
                if movement_distance > 0:
                    print(f"Movement: Move Forward by {movement_distance:.2f} meters")
                else:
                    print(f"Movement: Move Backward by {abs(movement_distance):.2f} meters")
            else:
                print("Movement: Stop")

        return {
            "distance": distance,
            "movement_distance": movement_distance,
            "angle_offset": angle_offset,
        }

    def start_tracking(self, print_type="movement"):
        while True:
            # only proceed if enough time has passed since the last poll
            current_time = time.time()
            if current_time - self.last_poll_time < self.polling_interval:
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("Could not read frame.")
                break

            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * frame_height
                hip_y = (left_hip.y + right_hip.y) / 2 * frame_height
                vertical_height = abs(shoulder_y - hip_y)
                user_center_x = (left_shoulder.x + right_shoulder.x) / 2 * frame_width
                horizontal_offset = user_center_x - frame_width / 2  

                distance = self.estimate_distance(vertical_height)
                movement_output = self.control_movement(distance, horizontal_offset, frame_width, print_type=print_type)

                # reset poll time after each update
                self.last_poll_time = current_time

                # display feedback on video frame
                text_color = (0, 255, 255)
                font_scale = 0.5
                cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
                cv2.putText(frame, f"Move Dist.: {movement_output['movement_distance']:.2f} m", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
                cv2.putText(frame, f"User Pos: ({int(user_center_x)}, {int(shoulder_y)})", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
                cv2.putText(frame, f"Angle Offset: {movement_output['angle_offset']:.2f} deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

                cv2.line(frame, 
                         (int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1]), int(shoulder_y)),
                         (int((left_hip.x + right_hip.x) / 2 * frame.shape[1]), int(hip_y)),
                         (255, 0, 0), 2)

            cv2.imshow("Distance Estimator", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Run the distance estimator
# before run6ning you can also adjust parameters in the class init
if __name__ == "__main__":
    estimator = DistanceAngleController()
    estimator.set_reference_distance()
    estimator.start_tracking(print_type="movement")  # choose "pwm" or "movement"
