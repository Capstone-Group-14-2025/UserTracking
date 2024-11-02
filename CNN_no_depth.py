import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import torch
import time

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize Tkinter window
    window = tk.Tk()
    window.title("Real-Time Object Detection")
    window.geometry("800x600")  # Adjust based on your preference

    # Create a label to display the video frames
    video_label = tk.Label(window)
    video_label.pack()

    # Load the pre-trained YOLOv8 model
    try:
        model = YOLO('yolov8s.pt')  # Switched to 'yolov8s.pt' for better accuracy
        model.to(device)  # Move model to GPU if available
        print("YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        return

    # Initialize video capture with the correct index (1 for B525 HD Webcam)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open the B525 HD Webcam.")
        return
    else:
        print("B525 HD Webcam opened successfully.")

    # Set webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced width for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced height for faster processing

    # Initialize FPS calculation
    prev_time = time.time()

    def update_frame():
        nonlocal prev_time
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            window.after(20, update_frame)  # Retry after a short delay
            return

        # Convert the frame to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection, filtering to only detect 'person' (class 0)
        try:
            results = model(rgb_frame, classes=[0], device=device, verbose=False)
            annotated_frame = results[0].plot()
        except Exception as e:
            print(f"Error during model prediction: {e}")
            window.after(20, update_frame)
            return

        # Convert the annotated frame to PIL Image
        pil_image = Image.fromarray(annotated_frame)

        # Resize the image to fit the Tkinter window (optional)
        try:
            pil_image = pil_image.resize((800, 600), Image.Resampling.LANCZOS)
        except AttributeError:
            pil_image = pil_image.resize((800, 600), Image.ANTIALIAS)  # For older Pillow versions

        # Convert PIL Image to ImageTk
        imgtk = ImageTk.PhotoImage(image=pil_image)

        # Update the label with the new image
        video_label.imgtk = imgtk  # Keep a reference to prevent garbage collection
        video_label.configure(image=imgtk)

        # Calculate and display FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        window.title(f"Real-Time Object Detection - FPS: {fps:.2f}")
        prev_time = current_time

        # Schedule the next frame update
        window.after(20, update_frame)  # Adjust the delay as needed (milliseconds)

    def on_closing():
        print("Closing the application...")
        cap.release()
        window.destroy()

    # Handle window closing
    window.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the frame update loop
    update_frame()

    # Start the Tkinter event loop
    window.mainloop()

if __name__ == "__main__":
    main()
