# README.txt for Tracking Code

## Distance Angle Controller (User Tracker)

This program is designed to track a user in real-time using video input and MediaPipe's pose detection. It calculates the user's distance and angular offset relative to the robot and provides control outputs to adjust the robot's position accordingly.

### Features
1. **Real-Time User Tracking**:
   - Detects user position using pose landmarks (shoulders and hips).
   - Measures shoulder-to-hip height for distance estimation.

2. **Distance and Angular Control**:
   - Maintains a target distance from the user.
   - Aligns robot direction based on angular offset.

3. **Configurable Outputs**:
   - Outputs as either Pulse Width Modulation (PWM) signals or descriptive movement instructions.

4. **Adjustable Parameters**:
   - Target distance, distance tolerance, angle tolerance, and polling interval are customizable.

5. **Calibration**:
   - Calibrates reference height for distance measurement based on a known reference distance.

### Requirements
- **Software**:
  - Python 3.x
  - OpenCV (`cv2`)
  - MediaPipe (`mediapipe`)
  - PySerial (`serial`)
  
- **Hardware**:
  - Webcam for video input.
  - Serial device (e.g., motor controller) for control signals.
  - Compatible robot chassis (e.g., Jaguar v2).

### How to Use
1. **Setup**:
   - Connect a camera to your computer.
   - Ensure the serial device (e.g., motor controller) is connected and configured properly.
   - Install required Python libraries: `pip install opencv-python mediapipe pyserial`.

2. **Run the Program**:
   - Execute the script: `python <script_name>.py`.
   - Calibrate the reference height by standing at a known distance and pressing `c`.
   - The program will track the user and maintain the desired distance and alignment.

3. **Controls**:
   - Press `c` to recalibrate the reference height.
   - Press `q` to quit the program.

4. **Outputs**:
   - Choose between:
     - `"pwm"`: Sends PWM signals to control the robot.
     - `"movement"`: Prints movement instructions (e.g., "Move Forward").

### Key Parameters
- `target_distance`: Desired distance to maintain from the user (in meters).
- `distance_tolerance`: Acceptable deviation from the target distance (in meters).
- `angle_tolerance`: Acceptable angular deviation (in degrees).
- `polling_interval`: Interval for processing frames (in seconds).

### Example
After calibration, the robot will:
- Move forward or backward to maintain the target distance.
- Rotate left or right to align with the user based on angular offset.

### Notes
- Ensure proper lighting for accurate pose detection.
- The serial device must be correctly configured for PWM signal output.

### Known Issues
- Pose detection accuracy may vary with occlusions or rapid movements.
- Distance estimation depends on consistent calibration.

### License
This code is provided as-is without warranty. Modify and use it according to your project's requirements.
