import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import torch
import time
import math
import numpy as np
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

class DistanceEstimatorApp:
    def __init__(self, window, window_title, video_source=1):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x600")
        
        # Initialize variables
        self.prev_time = time.time()
        self.scaling_factor = 4000.0  # Adjusted based on calibration
        
        # Setup video capture
        self.cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Could not open the webcam.")
            exit()
        else:
            print("Webcam opened successfully.")
        
        # Set webcam resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced width for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced height for faster processing
        
        # Initialize YOLOv8 model
        try:
            self.model = YOLO('yolov8s.pt')  # Use 'yolov8s.pt' for better accuracy
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            print(f"YOLOv8 model loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            exit()
        
        # Initialize DPT model and feature extractor
        try:
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            self.depth_model.to(self.device)
            self.depth_model.eval()
            print(f"DPT model loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading DPT model: {e}")
            exit()
        
        # Create a label to display the video frames
        self.video_label = tk.Label(window)
        self.video_label.pack()
        
        # Start the update loop
        self.update_frame()
        
        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
    
    def estimate_depth(self, image):
        # Convert image to PIL format
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs for the model
        inputs = self.feature_extractor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass to get depth
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Resize depth to original image size
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy array
        depth = predicted_depth.cpu().numpy()
        
        # Handle negative values by setting them to a minimal positive value
        depth[depth < 0] = 0.1  # Assign a minimal depth value to avoid division by zero
        
        return depth
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame.")
            self.window.after(20, self.update_frame)
            return
        
        try:
            results = self.model(frame, classes=[0], device=self.device, verbose=False)  # Class 0 is 'person'
            annotated_frame = results[0].plot()
            
            # Perform depth estimation using DPT
            depth_map = self.estimate_depth(frame)
            
            # Log depth map statistics
            print(f"Depth Map Stats - Min: {depth_map.min()}, Max: {depth_map.max()}, Mean: {depth_map.mean()}")
            
            # For each detected person, calculate distance
            boxes = results[0].boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_height = y2 - y1  # Height in pixels
                
                # Determine the center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Ensure center coordinates are within image boundaries
                center_x = min(max(center_x, 0), frame.shape[1] - 1)
                center_y = min(max(center_y, 0), frame.shape[0] - 1)
                
                # Log center coordinates
                print(f"Bounding Box Center - X: {center_x}, Y: {center_y}")
                
                # Get depth value at the center point
                depth_value = float(depth_map[center_y, center_x])
                
                # Log depth value
                print(f"Depth Value at Center: {depth_value}")
                
                # Prevent division by zero
                if depth_value == 0:
                    distance = 0.0
                else:
                    distance = float(self.scaling_factor) / depth_value  # Use '/' if higher depth_value means closer
                
                # Round the distance
                distance = round(distance-1, 2)
                
                # Log calculated distance
                print(f"Calculated Distance: {distance}m")
                
                # Define text properties
                text = f"{distance}m"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (255, 255, 255)  # White color for text
                thickness = 2
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Calculate position: inside the bounding box, near the bottom
                text_x = int(x1)
                text_y = int(y2) - 10  # 10 pixels above the bottom
                
                # Ensure the text does not go above the top of the bounding box
                if text_y - text_height < y1:
                    text_y = y1 + text_height + 10  # Adjust if necessary
                
                # Draw a filled rectangle for better text visibility
                cv2.rectangle(annotated_frame, 
                              (text_x, text_y - text_height - 5), 
                              (text_x + text_width, text_y + 5), 
                              (0, 0, 0), 
                              cv2.FILLED)
                
                # Put text over the rectangle
                cv2.putText(annotated_frame, text, 
                            (text_x, text_y), 
                            font, 
                            font_scale, 
                            color, 
                            thickness)
                    
        except Exception as e:
            print(f"Error during model prediction: {e}")
            self.window.after(20, self.update_frame)
            return
        
        # Convert the annotated frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        # Resize the image to fit the Tkinter window
        pil_image = pil_image.resize((800, 600), Image.ANTIALIAS)
        
        # Convert PIL Image to ImageTk
        imgtk = ImageTk.PhotoImage(image=pil_image)
        
        # Update the label with the new image
        self.video_label.imgtk = imgtk  # Keep a reference to prevent garbage collection
        self.video_label.configure(image=imgtk)
        
        # Calculate and display FPS
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        self.window.title(f"Real-Time Object Detection with Depth - FPS: {fps:.2f}")
        self.prev_time = current_time
        
        # Schedule the next frame update
        self.window.after(20, self.update_frame)  # Adjust the delay as needed (milliseconds)
    
    def on_closing(self):
        print("Closing the application...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DistanceEstimatorApp(root, "Real-Time Object Detection with Depth Estimation")
