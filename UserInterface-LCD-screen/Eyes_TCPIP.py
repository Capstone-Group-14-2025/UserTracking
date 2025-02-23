#!/usr/bin/env python3

import sys
import tkinter as tk
import random
import math
import threading
import queue  # For thread-safe queues
import socket

# Directions
DIRECTION_STRAIGHT = "straight"
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"

# Map direction strings to a numerical offset
#  -1.0 = fully left, 0.0 = center, 1.0 = fully right
DIRECTION_OFFSETS = {
    DIRECTION_LEFT: -1.0,
    DIRECTION_STRAIGHT: 0.0,
    DIRECTION_RIGHT: 1.0
}

# Emotions (colors) we can fade between. Chosen arbitrarily; adjust as needed.
# We'll interpret each emotion as a single "base color" for the outer eye shape.
# Then we do a color fade from old to new. Once done, we switch the lens shape.
EMOTIONS = {
    "neutral": {
        "outer": "#3399FF",  # Vibrant blue
        "lens":  "#70CFFF"   # Lighter blue
    },
    "angry": {
        "outer": "#E63737",  # Bright red
        "lens":  "#FFA3A3"   # Lighter red
    },
    "happy": {
        "outer": "#2CE82C",  # Bright green
        "lens":  "#CFFFc4"   # Lighter green
    }
}

def hex_to_rgb(hex_color):
    """
    Convert a hex color string to an (R, G, B) tuple.
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Input #{hex_color} is not in #RRGGBB format.")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_tuple):
    """
    Convert an (R, G, B) tuple to a hex color string.
    """
    return "#{:02X}{:02X}{:02X}".format(*rgb_tuple)

# Initialize a thread-safe queue for commands
command_queue = queue.Queue()

def command_input_thread(cmd_queue):
    """
    Thread function to read commands from stdin and put them into the queue.
    """
    while True:
        try:
            # Read a line from stdin
            line = sys.stdin.readline()
            if not line:
                break  # EOF reached
            line = line.strip().lower()
            if line:
                cmd_queue.put(line)
        except Exception as e:
            print(f"Error reading stdin: {e}")
            break

# ------------------------------------------------------------------------------
# Simple TCP Server to receive commands
# ------------------------------------------------------------------------------
def start_tcp_server(cmd_queue, host, port):
    """
    Starts a basic TCP server in a background thread.
    Each connected client can send commands like "left", "right", "blink", etc.
    Commands are enqueued into the command_queue.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"[Main TCP Server] Listening on {host}:{port}...")

    def handle_client(client_conn, client_addr):
        print(f"[Main TCP Server] Connection from {client_addr}")
        with client_conn:
            while True:
                try:
                    data = client_conn.recv(1024)
                    if not data:
                        break  # client closed
                    lines = data.decode("utf-8").splitlines()
                    for line in lines:
                        line = line.strip().lower()
                        if line:
                            cmd_queue.put(line)
                except Exception as e:
                    print(f"[Main TCP Server] Error handling client {client_addr}: {e}")
                    break
        print(f"[Main TCP Server] Client {client_addr} disconnected.")

    def accept_loop():
        while True:
            try:
                conn, addr = server_socket.accept()
                # Handle each client in a separate thread
                t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                t.start()
            except OSError:
                # Socket likely closed if the app is shutting down
                break

    # Start the accept() loop in a background thread
    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()


class RoboEyesApp:
    def __init__(self, root, width, height, fps, cmd_queue):
        self.root = root
        self.root.title("RoboEyes App")

        self.width = width
        self.height = height
        # Remove borders and highlight thickness from the canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.width, 
            height=self.height, 
            bg="black",
            highlightthickness=0,  # Remove highlight border
            bd=0  # Remove border
        )
        self.canvas.pack()

        # Eye geometry
        self.eye_width = 300
        self.eye_height = 450

        # Eye positions
        self.left_eye_x = (self.width // 2) - (self.eye_width + 40)
        self.left_eye_y = (self.height // 2) - (self.eye_height // 2)
        self.right_eye_x = (self.width // 2) + 40
        self.right_eye_y = self.left_eye_y

        # Current direction is stored as a float offset in [-1..1]
        self.current_dir_offset = 0.0
        # Target direction offset we want to reach (for smooth transitions)
        self.target_dir_offset = 0.0

        # Current emotion colors as (R, G, B)
        self.current_emotion_color = hex_to_rgb(EMOTIONS["neutral"]["outer"])
        self.current_lens_color = hex_to_rgb(EMOTIONS["neutral"]["lens"])
        # The actual "shape" to draw
        self.current_emotion = "neutral"
        # Pending emotion to switch to after transition
        self.pending_emotion = "neutral"

        # Eye “open/closed” states
        self.left_eye_open = True
        self.right_eye_open = True
        self.blink_in_progress = False

        # Animation / framerate
        self.fps = fps
        self.frame_interval = int(1000 / self.fps)

        # Command queue
        self.cmd_queue = cmd_queue

        # Kick off the update & blinking loops
        self.update_eyes()
        self.schedule_random_blink()
        self.process_commands()

    # ---------------------
    # Public Setters
    # ---------------------
    def set_direction(self, direction):
        """Set the target direction for smooth transition."""
        if direction not in DIRECTION_OFFSETS:
            print(f"Direction '{direction}' not recognized.")
            return
        self.target_dir_offset = DIRECTION_OFFSETS[direction]
        print(f"Set target direction to {direction} ({self.target_dir_offset})")

    def set_emotion(self, emotion):
        """Set the target emotion with color transition."""
        if emotion not in EMOTIONS:
            print(f"Emotion '{emotion}' not recognized.")
            return
        self.pending_emotion = emotion
        target_outer = hex_to_rgb(EMOTIONS[emotion]["outer"])
        target_lens = hex_to_rgb(EMOTIONS[emotion]["lens"])
        self.target_emotion_color = target_outer
        self.target_lens_color = target_lens
        print(f"Set target emotion to {emotion}")

    def blink(self):
        """Close eyes briefly, then reopen."""
        if not self.blink_in_progress:
            self.blink_in_progress = True
            self.left_eye_open = False
            self.right_eye_open = False
            self.root.after(50, self.end_blink)  # Blink duration

    def end_blink(self):
        self.left_eye_open = True
        self.right_eye_open = True
        self.blink_in_progress = False

    # ---------------------
    # Main Animation Loop
    # ---------------------
    def update_eyes(self):
        """Update current direction & emotion transitions, then draw everything. Re-scheduled each frame."""
        self.update_direction_transition()
        self.update_emotion_transition()

        # Clear the canvas
        self.canvas.delete("all")

        # Draw both eyes
        self.draw_eye(
            x=self.left_eye_x,
            y=self.left_eye_y,
            w=self.eye_width,
            h=self.eye_height,
            open_=self.left_eye_open,
            direction_offset=self.current_dir_offset if not self.blink_in_progress else 0.0,
            outer_color=self.current_emotion_color,
            lens_color=self.current_lens_color,
            emotion_shape=self.current_emotion
        )
        self.draw_eye(
            x=self.right_eye_x,
            y=self.right_eye_y,
            w=self.eye_width,
            h=self.eye_height,
            open_=self.right_eye_open,
            direction_offset=self.current_dir_offset if not self.blink_in_progress else 0.0,
            outer_color=self.current_emotion_color,
            lens_color=self.current_lens_color,
            emotion_shape=self.current_emotion
        )

        self.root.after(self.frame_interval, self.update_eyes)

    # ---------------------
    # Direction Transition
    # ---------------------
    def update_direction_transition(self):
        """
        Move 'current_dir_offset' slowly toward 'target_dir_offset'.
        You can adjust the speed by changing 'step'.
        """
        step = 0.06  # how quickly we move per frame
        diff = self.target_dir_offset - self.current_dir_offset
        if abs(diff) < 0.01:
            # Close enough, just set it
            self.current_dir_offset = self.target_dir_offset
        else:
            # Move by a small fraction
            self.current_dir_offset += step * (1 if diff > 0 else -1)

    # ---------------------
    # Emotion Transition
    # ---------------------
    def update_emotion_transition(self):
        """
        Fade current_emotion_color to target_emotion_color.
        Once fully reached, change the 'current_emotion' shape.
        """
        if not hasattr(self, 'target_emotion_color'):
            return  # No target set yet

        (r0, g0, b0) = self.current_emotion_color
        (r1, g1, b1) = self.target_emotion_color

        # Speed: the bigger the step, the faster the color changes
        fade_step = 6

        new_r = self._approach(r0, r1, fade_step)
        new_g = self._approach(g0, g1, fade_step)
        new_b = self._approach(b0, b1, fade_step)

        self.current_emotion_color = (new_r, new_g, new_b)
        # Update lens color similarly
        (lr0, lg0, lb0) = self.current_lens_color
        (lr1, lg1, lb1) = self.target_lens_color

        new_lr = self._approach(lr0, lr1, fade_step)
        new_lg = self._approach(lg0, lg1, fade_step)
        new_lb = self._approach(lb0, lb1, fade_step)

        self.current_lens_color = (new_lr, new_lg, new_lb)

        # If we've arrived at the target color, switch shapes
        if (self.current_emotion_color == self.target_emotion_color and
            self.current_lens_color == self.target_lens_color):
            if self.current_emotion != self.pending_emotion:
                self.current_emotion = self.pending_emotion
                print(f"Emotion transitioned to {self.current_emotion}")

    def _approach(self, cur, tgt, step):
        """Helper to move cur -> tgt by up to 'step', clamped so we don't overshoot."""
        if cur < tgt:
            return min(cur + step, tgt)
        elif cur > tgt:
            return max(cur - step, tgt)
        else:
            return cur

    # ---------------------
    # Eye Drawing
    # ---------------------
    def draw_eye(self, x, y, w, h, open_, direction_offset, outer_color, lens_color, emotion_shape):
        """
        - direction_offset ∈ [-1..1]: shift pupil left or right.
        - outer_color: an (R,G,B) tuple for the outer shape color.
        - lens_color: an (R,G,B) tuple for the lens color.
        - emotion_shape: which shape to draw inside (angry / happy / neutral).
        """
        # If the eye is closed, drastically shrink the vertical dimension
        if not open_:
            h = max(8, h // 8)

        # Build the color strings
        outer_color_str = rgb_to_hex(outer_color)
        lens_color_str = rgb_to_hex(lens_color)

        # Outer shape (rounded rect)
        corner_radius = min(w, h) // 5
        self.draw_rounded_rect(x, y, x + w, y + h, corner_radius,
                               fill=outer_color_str, outline=outer_color_str)

        # Lens bounding box
        lens_padding = w // 8
        lens_w = w - 2 * lens_padding
        lens_h = h - 2 * lens_padding
        lens_x = x + lens_padding
        lens_y = y + lens_padding

        # Shift if looking left or right, scaled by direction_offset
        lens_x += int(direction_offset * lens_padding)

        # If eye is "closed", shrink lens vertically
        if not open_:
            lens_h = max(4, lens_h // 4)
            lens_y = y + (h // 2) - (lens_h // 2)

        # Draw the lens based on shape
        if emotion_shape == "angry":
            self._draw_angry_lens(lens_x, lens_y, lens_w, lens_h, lens_color_str)
        elif emotion_shape == "happy":
            self._draw_happy_lens(lens_x, lens_y, lens_w, lens_h, lens_color_str)
        else:
            # default or "neutral"
            self._draw_neutral_lens(lens_x, lens_y, lens_w, lens_h, lens_color_str)

    def _draw_angry_lens(self, lx, ly, lw, lh, color):
        # Spiky polygon
        lens_points = [
            (lx,        ly),
            (lx + lw,   ly),
            (lx + lw,   ly + lh),
            (lx,        ly + lh),
            (lx + lw//2,ly + lh//2)
        ]
        # Flatten the list of tuples
        flat_points = [coord for point in lens_points for coord in point]
        self.canvas.create_polygon(*flat_points, fill=color, outline=color)
        # Optional: Add flicker effect
        # self.draw_polygon_flicker(lens_points)

    def _draw_happy_lens(self, lx, ly, lw, lh, color):
        # Half-ellipse
        num_points = 100
        lens_points = []
        rx = lw / 2
        ry = lh
        cx = lx + rx
        cy = ly + lh

        for i in range(num_points + 1):
            theta = math.pi * i / num_points
            x = cx + rx * math.cos(theta)
            y = cy - ry * math.sin(theta)
            lens_points.append((x, y))

        # Close the bottom
        lens_points.append((lx + lw, ly + lh))
        lens_points.append((lx,      ly + lh))

        # Flatten the list of tuples
        flat_points = [coord for point in lens_points for coord in point]
        self.canvas.create_polygon(*flat_points, fill=color, outline=color)
        # Optional: Add flicker effect
        # self.draw_polygon_flicker(lens_points)

    def _draw_neutral_lens(self, lx, ly, lw, lh, color):
        lens_points = [
            (lx,      ly),
            (lx + lw, ly),
            (lx + lw, ly + lh),
            (lx,      ly + lh)
        ]
        # Flatten the list of tuples
        flat_points = [coord for point in lens_points for coord in point]
        self.canvas.create_polygon(*flat_points, fill=color, outline=color)
        # Optional: Add flicker effect
        # self.draw_polygon_flicker(flat_points)

    # --------------------------------------------------------------------------
    # Flicker-Scan-line Logic (Optional)
    # --------------------------------------------------------------------------
    def draw_polygon_flicker(self, points):
        """
        Fill 'points' polygon with horizontal lines in random tinted colors.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        line_spacing = 4
        for row in range(int(min_y), int(max_y)+1, line_spacing):
            inter_xs = self.polygon_row_intersections(points, row)
            if len(inter_xs) < 2:
                continue
            inter_xs.sort()

            # Pair up intersection points
            for i in range(0, len(inter_xs)-1, 2):
                x1 = inter_xs[i]
                x2 = inter_xs[i+1]
                shade = random.randint(0, 15)
                g_base = 0x22
                b_base = 0x44
                g = min(0xFF, g_base + shade)
                b = min(0xFF, b_base + shade)
                color = f"#00{g:02x}{b:02x}"
                self.canvas.create_line(x1, row, x2, row, fill=color, width=1)

    def polygon_row_intersections(self, points, row):
        inter_xs = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i+1) % n]

            if y1 == y2:
                continue
            if not (min(y1, y2) <= row <= max(y1, y2)):
                continue

            t = (row - y1) / float(y2 - y1)
            if 0 <= t <= 1:
                x_inter = x1 + t*(x2 - x1)
                inter_xs.append(x_inter)
        return inter_xs

    # --------------------------------------------------------------------------
    # Outer Eye Shape (unchanged)
    # --------------------------------------------------------------------------
    def draw_rounded_rect(self, x1, y1, x2, y2, r, fill="", outline=""):
        r = min(r, (x2 - x1)//2, (y2 - y1)//2)
        self.canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=fill, outline=outline)
        self.canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=fill, outline=outline)
        self.canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=fill, outline=outline)
        self.canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=fill, outline=outline)
        self.canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=fill, outline=outline)
        self.canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=fill, outline=outline)

    # --------------------------------------------------------------------------
    # Random Blinking (unchanged)
    # --------------------------------------------------------------------------
    def schedule_random_blink(self):
        interval_ms = random.uniform(0.3, 2.0) * 1000  # Further reduced interval for even faster blinking
        self.root.after(int(interval_ms), self._random_blink)

    def _random_blink(self):
        if not self.blink_in_progress:
            self.blink()
        self.schedule_random_blink()

    # ---------------------
    # Command Processing
    # ---------------------
    def process_commands(self):
        """
        Check the command queue and process any pending commands.
        """
        while not self.cmd_queue.empty():
            try:
                cmd = self.cmd_queue.get_nowait()
            except queue.Empty:
                break  # No more commands

            if cmd in ["quit", "exit"]:
                print("Exiting on command.")
                self.root.quit()
                return
            elif cmd in [DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_STRAIGHT]:
                self.set_direction(cmd)
                print(f"Set direction to {cmd}")
            elif cmd == "blink":
                self.blink()
                print("Blinking!")
            elif cmd in ["happy", "angry", "neutral"]:
                self.set_emotion(cmd)
                print(f"Emotion set to {cmd}")
            else:
                print("Unknown command:", cmd)

        # Schedule the next command check
        self.root.after(50, self.process_commands)  # Check every 100 ms

def main():

    print("Available Commands: left, right, straight, blink, happy, angry, neutral, quit/exit")

    root = tk.Tk()

    # Remove window decorations (border, title bar, etc.)
    root.overrideredirect(True)

    # Make the window full screen using the -fullscreen attribute
    root.attributes("-fullscreen", True)

    # Hide the mouse cursor for the root window
    root.config(cursor="none")

    # Bind the Escape key to exit full screen
    root.bind('<Escape>', lambda e: root.destroy())

    # Start the main TCP server (listens on port 12345)
    start_tcp_server(command_queue, host='0.0.0.0', port=12345)

    # Start the command input thread for stdin
    # If you want to disable stdin commands, comment out the following two lines
    input_thread = threading.Thread(target=command_input_thread, args=(command_queue,), daemon=True)
    input_thread.start()

    # Create the RoboEyesApp with the command queue
    app = RoboEyesApp(
        root,
        width=root.winfo_screenwidth(),
        height=root.winfo_screenheight(),
        fps=120,  # Further increased FPS for even smoother and faster animations
        cmd_queue=command_queue  # Pass the command queue
    )

    # Additionally, hide the cursor on the canvas
    app.canvas.config(cursor="none")

    # Start the Tk event loop
    root.mainloop()
    print("GUI closed. Exiting...")

if __name__ == "__main__":
    main()
