#!/usr/bin/env python3

import sys
import select
import tkinter as tk
import random
import math

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

# Emotions (colors) we can fade between. Chosen arbitrary; adjust as needed.
# We'll interpret each emotion as a single "base color" for the outer eye shape.
# Then we do a color fade from old to new. Once done, we switch the lens shape.
EMOTIONS = {
    "neutral": (0x00, 0x22, 0x55),  # Dark bluish
    "angry":   (0x60, 0x00, 0x00),  # Dark red
    "happy":   (0x00, 0x44, 0x00)   # Dark green
}


class RoboEyesApp:
    def __init__(self, root, width=600, height=300, fps=30):
        self.root = root
        self.root.title("Transitions Demo")

        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        # Eye geometry
        self.eye_width = 120
        self.eye_height = 160

        # Eye positions
        self.left_eye_x = (self.width // 2) - (self.eye_width + 20)
        self.left_eye_y = (self.height // 2) - (self.eye_height // 2)
        self.right_eye_x = (self.width // 2) + 20
        self.right_eye_y = self.left_eye_y

        # Current direction is stored as a float offset in [-1..1]
        self.current_dir_offset = 0.0
        # Target direction offset we want to reach
        self.target_dir_offset = 0.0

        # Current emotion color as an (R,G,B)
        self.current_emotion_color = EMOTIONS["neutral"]
        # Target color we want to fade to
        self.target_emotion_color = EMOTIONS["neutral"]
        # The actual "shape" to draw. We'll keep track separately
        # so that during color fade, we haven't fully switched shapes yet.
        self.current_emotion = "neutral"
        # We'll store a "pending_emotion" that we'll switch to once color fade completes
        self.pending_emotion = "neutral"

        # Eye “open/closed” states
        self.left_eye_open = True
        self.right_eye_open = True
        self.blink_in_progress = False

        # Animation / framerate
        self.fps = fps
        self.frame_interval = int(1000 / self.fps)

        # Kick off the update & blinking loops
        self.update_eyes()
        self.schedule_random_blink()

    # ---------------------
    # Public Setters
    # ---------------------
    def set_direction(self, direction):
        """Requests a new direction by setting the target offset."""
        if direction not in DIRECTION_OFFSETS:
            return
        self.target_dir_offset = DIRECTION_OFFSETS[direction]

    def set_emotion(self, emotion):
        """Requests a new emotion by setting the target color; shape will change when fade completes."""
        if emotion not in EMOTIONS:
            return
        self.target_emotion_color = EMOTIONS[emotion]
        self.pending_emotion = emotion

    def blink(self):
        """Close eyes briefly, then reopen."""
        if not self.blink_in_progress:
            self.blink_in_progress = True
            self.left_eye_open = False
            self.right_eye_open = False
            self.root.after(150, self.end_blink)

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
            emotion_color=self.current_emotion_color,
            emotion_shape=self.current_emotion
        )
        self.draw_eye(
            x=self.right_eye_x,
            y=self.right_eye_y,
            w=self.eye_width,
            h=self.eye_height,
            open_=self.right_eye_open,
            direction_offset=self.current_dir_offset if not self.blink_in_progress else 0.0,
            emotion_color=self.current_emotion_color,
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
        # Unpack current and target
        (r0, g0, b0) = self.current_emotion_color
        (r1, g1, b1) = self.target_emotion_color

        # Speed: the bigger the step, the faster the color changes
        fade_step = 6

        # Compute new color each frame
        new_r = self._approach(r0, r1, fade_step)
        new_g = self._approach(g0, g1, fade_step)
        new_b = self._approach(b0, b1, fade_step)

        self.current_emotion_color = (new_r, new_g, new_b)

        # If we've arrived at the target color, switch shapes
        if (new_r, new_g, new_b) == (r1, g1, b1):
            self.current_emotion = self.pending_emotion

    def _approach(self, cur, tgt, step):
        """
        Helper to move cur -> tgt by up to 'step', clamped so we don't overshoot.
        """
        if cur < tgt:
            return min(cur + step, tgt)
        elif cur > tgt:
            return max(cur - step, tgt)
        else:
            return cur

    # ---------------------
    # Eye Drawing
    # ---------------------
    def draw_eye(self, x, y, w, h, open_, direction_offset, emotion_color, emotion_shape):
        """
        - direction_offset ∈ [-1..1]: shift pupil left or right.
        - emotion_color: an (R,G,B) tuple for the outer shape color.
        - emotion_shape: which shape to draw inside (angry / happy / neutral).
        """
        # If the eye is closed, drastically shrink the vertical dimension
        if not open_:
            h = max(8, h // 8)

        # Build the color strings (outer/lens).
        # We'll use the same base color for outer and shift it for the lens.
        base_color = "#{:02x}{:02x}{:02x}".format(*emotion_color)

        # Outer shape (rounded rect)
        corner_radius = min(w, h) // 5
        self.draw_rounded_rect(x, y, x + w, y + h, corner_radius,
                               fill=base_color, outline=base_color)

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

        # Now pick lens color, slightly lighter or different from base to show variety.
        # Just an example shift.
        lens_r = min(255, emotion_color[0] + 50)
        lens_g = min(255, emotion_color[1] + 50)
        lens_b = min(255, emotion_color[2] + 50)
        lens_color_str = "#{:02x}{:02x}{:02x}".format(lens_r, lens_g, lens_b)

        # Draw the lens based on shape
        if emotion_shape == "angry":
            self._draw_angry_lens(lens_x, lens_y, lens_w, lens_h, lens_color_str)
        elif emotion_shape == "happy":
            self._draw_happy_lens(lens_x, lens_y, lens_w, lens_h, lens_color_str)
        else:
            # default or "neutral"
            self._draw_neutral_lens(lens_x, lens_y, lens_w, lens_h, lens_color_str)

    def _draw_angry_lens(self, lx, ly, lw, lh, color):
        # spiky polygon
        lens_points = [
            (lx,        ly),
            (lx + lw,   ly),
            (lx + lw,   ly + lh),
            (lx,        ly + lh),
            (lx + lw//2,ly + lh//2)
        ]
        self.canvas.create_polygon(*lens_points, fill=color, outline=color)
        self.draw_polygon_flicker(lens_points)

    def _draw_happy_lens(self, lx, ly, lw, lh, color):
        # half-ellipse
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

        # close the bottom
        lens_points.append((lx + lw, ly + lh))
        lens_points.append((lx,      ly + lh))

        self.canvas.create_polygon(*lens_points, fill=color, outline=color)
        self.draw_polygon_flicker(lens_points)

    def _draw_neutral_lens(self, lx, ly, lw, lh, color):
        lens_points = [
            (lx,      ly),
            (lx + lw, ly),
            (lx + lw, ly + lh),
            (lx,      ly + lh)
        ]
        self.canvas.create_polygon(*lens_points, fill=color, outline=color)
        self.draw_polygon_flicker(lens_points)

    # --------------------------------------------------------------------------
    # Flicker-Scan-line Logic (unchanged)
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

            # pair up intersection points
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
            if not (min(y1,y2) <= row <= max(y1,y2)):
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
        interval_ms = random.uniform(1.0, 5.0) * 1000
        self.root.after(int(interval_ms), self._random_blink)

    def _random_blink(self):
        if not self.blink_in_progress:
            self.blink()
        self.schedule_random_blink()


def poll_console(app):
    """
    Non-blocking read from stdin using select.
    """
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip().lower()
        if not line:
            return

        if line in ["quit", "exit"]:
            print("Exiting on command.")
            app.root.quit()
            return
        elif line in [DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_STRAIGHT]:
            app.set_direction(line)
            print(f"Set direction to {line}")
        elif line == "blink":
            app.blink()
            print("Blinking!")
        elif line in ["happy", "angry", "neutral"]:
            app.set_emotion(line)
            print(f"Emotion set to {line}")
        else:
            print("Unknown command:", line)

    app.root.after(200, poll_console, app)


def main():
    print("Commands: left, right, straight, blink, happy, angry, neutral, quit/exit")
    root = tk.Tk()
    app = RoboEyesApp(root, width=900, height=450, fps=30)

    poll_console(app)
    root.mainloop()
    print("GUI closed. Exiting...")


if __name__ == "__main__":
    main()
