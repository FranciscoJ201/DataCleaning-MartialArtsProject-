import json, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from matplotlib.widgets import Button, Slider, TextBox
from JSON_FILES.JSONREAD import filecleanup, filecleanupsingle
from util import distTwoPoints,lim,SMPL24_EDGES,SMPL_BODY_PARTS # Import the distance function
from DataToJson import createSegmentJson

# =========================================================================
# 1. STANDALONE DATA LOADING HELPERS
# =========================================================================

def frame_number(k: str) -> int:
    base = os.path.splitext(k)[0]
    try:
        return int(base)
    except:
        # Fallback: grab last underscore-separated int
        for part in base.split('_')[::-1]:
            if part.isdigit():
                return int(part)
        return 0

def load_frames(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Group by image_id
    frames = {}
    for d in data:
        frames.setdefault(d["image_id"], []).append(d)
    # Sort by frame number
    keys = sorted(frames.keys(), key=frame_number)
    return keys, frames

def select_person_entry(entries, target_idx=None):
    if not entries:
        return None
    if target_idx is None:
        return entries[0]  # first person in frame
    for e in entries:
        if e.get("idx") == target_idx:
            return e
    return None  # not found this frame

def get_xyz_from_entry(entry):
    """
    Returns (x, y, z) arrays or (None, None, None) if missing.
    Expects 'pred_xyz_jts' shape (J, 3).
    """
    if entry is None:
        return None, None, None
    kj = entry.get("pred_xyz_jts")
    if kj is None:
        return None, None, None
    kp = np.array(kj, dtype=float)
    if kp.ndim != 2 or kp.shape[1] < 3:
        return None, None, None
    return kp[:, 0], kp[:, 1], kp[:, 2]


# =========================================================================
# 2. MAIN PLAYER CLASS
# =========================================================================

class Pose3DPlayer:
    # ---------------------------------------------------------------------
    # 2.1. INITIALIZATION AND STATE
    # ---------------------------------------------------------------------
    def __init__(
        self,
        json_path,
        target_idx=None,           # track id to follow, or None for first person
        edges=SMPL24_EDGES,
        fps=15,
        fixed_limits= lim,         # e.g., (-1000,1000) to force all axes same range
        auto_scale_margin=1.2,     # margin factor if not using fixed_limits
        point_size=40,
        sf_vertical = 0.0
    ):
        # --- Config & Data Loading ---
        self.keys, self.frames = load_frames(json_path)
        if not self.keys:
            raise RuntimeError("No frames found in JSON.")
        self.fps = max(1, int(fps))
        self.target_idx = target_idx
        self.edges = edges
        self.interval = int(1000 / self.fps)
        self.fixed_limits = fixed_limits
        self.auto_scale_margin = auto_scale_margin
        self.point_size = point_size
        self.json_path = json_path
        self.sf_vertical = sf_vertical

        # --- State Variables ---
        self.i = 0
        self.playing = False
        self.collected_data = [] # Format: [(frame_label, distance_in_inches), ...]
        self.selected_start = 0
        self.selected_end = len(self.keys) - 1
        
        # Custom connection (for user-selected points)
        self.custom_line_points = (0, 0) # INITIALIZED TO A DEFAULT
        
        # --- Figure & Axes Setup ---
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("3D Pose Player")

        # --- Initial Frame Data & Artists ---
        x, y, z = self._get_xyz(self.i)
        if x is None:
            # Try to find a frame with data
            for k in range(len(self.keys)):
                x, y, z = self._get_xyz(k)
                if x is not None:
                    self.i = k
                    break
        if x is None:
            raise RuntimeError("Could not find any frame with 'pred_xyz_jts' data.")

        # Main keypoint scatter plot
        self.scat = self.ax.scatter3D(x, y, z, s=self.point_size)
        
        # Standard connections (lines for SMPL edges)
        self.lines = []
        for (a, b) in self.edges:
            if a < len(x) and b < len(x):
                ln, = self.ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
                self.lines.append((ln, a, b))

        # Custom connection (for user-selected points)
        self.custom_line, = self.ax.plot([],[],[], color = 'red',linewidth=5)
        self.custom_line_label = self.ax.text(0,0,0,"", color="red",fontsize=12)

        # --- Axes Limits & Labels ---
        self._set_limits(x, y, z)

        # --- UI Setup (Widgets) ---
        self._add_widgets()

        # --- Event Bindings ---
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._on_timer)


    # ---------------------------------------------------------------------
    # 2.2. DATA ACCESS AND CALCULATION METHODS
    # ---------------------------------------------------------------------
    
    def connect_points(self,pointAIdx, pointBIdx):
        #sets the indices for a line to be drawn between keypoints
        self.custom_line_points = (pointAIdx,pointBIdx)
        self._draw_frame(self.i)

    def output_lengths(self, x, y, z, frame_label):
       
        """
        Calculates and stores the real-life distance for the custom selected
        keypoints in a single frame.
        """
        if not self.custom_line_points:
            return

        a, b = self.custom_line_points
        
        # Check if the keypoint indices are valid for the current data
        if a >= len(x) or b >= len(x):
            return

        try:
            # 1. Calculate Keypoint Distance (KP Units)
            kp_distance = distTwoPoints(x[a], y[a], z[a], x[b], y[b], z[b])
            
            # 2. Convert to Real-Life Distance (Inches)
            real_life_distance = kp_distance * self.sf_vertical
            
            # 3. Store the result
            self.collected_data.append((frame_label, real_life_distance))

            # 4. Optional: Print the result (useful for verification)
            print(f"[OUTPUT] Frame: {frame_label} | Distance ({a}-{b}): {real_life_distance:.2f} inches")

        except Exception as e:
            print(f"An unexpected error occurred during distance calculation for frame {frame_label}: {e}")

    def _collect_segment_distances(self): 
        """
        Iterates through the selected frame range and collects the keypoint 
        distance for the custom-selected points in each frame.
        """
        if not self.custom_line_points:
            print("[Distance Collection] No custom keypoints selected.")
            return

        self.collected_data = [] # Clear previous data before collecting a new range
        
        # Determine the valid range indices based on the self.keys list
        start_i = self._clamp_idx(self.selected_start)
        end_i = self._clamp_idx(self.selected_end)

        print(f"[Distance Collection] Processing frames from {start_i} to {end_i}...")

        # Loop from start_i up to and including end_i
        for i in range(start_i, end_i + 1):
            x, y, z = self._get_xyz(i)
            frame_label = self.keys[i]
            
            if x is not None:
                # Call the single-frame distance function
                self.output_lengths(x, y, z, frame_label)

    # --- Data helpers ---
    def _get_xyz(self, idx):
        key = self.keys[idx]
        entries = self.frames[key]
        entry = select_person_entry(entries, self.target_idx)
        return get_xyz_from_entry(entry)

    def _set_limits(self, x, y, z):
        if self.fixed_limits is not None:
            lo, hi = self.fixed_limits
            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(lo, hi)
            self.ax.set_zlim(lo, hi)
        else:
            # autoscale with margin (makes the box cubic-ish)
            xs = np.array(x); ys = np.array(y); zs = np.array(z)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            zmin, zmax = zs.min(), zs.max()

            cmin = min(xmin, ymin, zmin)
            cmax = max(xmax, ymax, zmax)
            span = (cmax - cmin) * self.auto_scale_margin
            center = (cmax + cmin) / 2.0
            lo = center - span / 2.0
            hi = center + span / 2.0

            self.ax.set_xlim(lo, hi)
            self.ax.set_ylim(lo, hi)
            self.ax.set_zlim(lo, hi)
        
        # Set axis labels
        self.ax.set_xlabel("Left-Right Y")
        self.ax.set_ylabel("Up-Down Z")
        self.ax.set_zlabel("Forward-Back X")

    # ---------------------------------------------------------------------
    # 2.3. UI AND DRAWING METHODS
    # ---------------------------------------------------------------------

    def _add_widgets(self):
        # Make extra room at the bottom
        # Increased bottom margin to accommodate the new input box
        plt.subplots_adjust(bottom=0.35) 

        # Button axes (Prev/Play/Next/FPS - Moved up)
        ax_prev   = plt.axes([0.12, 0.25, 0.10, 0.06])
        ax_play   = plt.axes([0.24, 0.25, 0.12, 0.06])
        ax_next   = plt.axes([0.38, 0.25, 0.10, 0.06])
        ax_fps    = plt.axes([0.55, 0.25, 0.25, 0.06])

        # Row 1: Range text inputs (Start/End)
        ax_start  = plt.axes([0.12, 0.17, 0.18, 0.06])
        ax_end    = plt.axes([0.33, 0.17, 0.18, 0.06])
        ax_output_name = plt.axes([0.6, 0.17, 0.25, 0.06])
        # Row 2: NEW Keypoint Input & Save Button
        ax_kp_input = plt.axes([0.12, 0.09, 0.25, 0.06])
        ax_save     = plt.axes([0.40, 0.09, 0.15, 0.06]) # Dedicated save button

        # Slider axes (Moved down)
        ax_slider = plt.axes([0.12, 0.02, 0.76, 0.04])

        # Buttons
        self.btn_prev   = Button(ax_prev, "Prev")
        self.btn_play   = Button(ax_play, "Play")
        self.btn_next   = Button(ax_next, "Next")
        
        # Dedicated save button
        self.btn_save_points = Button(ax_save, "Save Points")

        # Text Boxes
        self.tb_start = TextBox(ax_start, "Start", initial=str(self.selected_start))
        self.tb_end   = TextBox(ax_end,   "End",   initial=str(self.selected_end))
        self.tb_output_name = TextBox(ax_output_name, "File Name", initial="output_data.json")
        # NEW TEXTBOX FOR KEYPOINT INPUT
        start_kp, end_kp = self.custom_line_points if self.custom_line_points else (0, 0)
        self.tb_new_points = TextBox(ax_kp_input, "Points (A,B)", initial=f"{start_kp},{end_kp}")

        # Sliders
        self.slider     = Slider(ax_slider, "Frame", 0, len(self.keys) - 1, valinit=self.i, valstep=1)
        self.fps_slider = Slider(ax_fps, "FPS", 1, 60, valinit=self.fps, valstep=1)

        # Camera preset
        self.ax.view_init(elev=110, azim=90)

        # Wire up controls
        self.btn_prev.on_clicked(lambda evt: self.step(-1))
        self.btn_next.on_clicked(lambda evt: self.step(1))
        self.btn_play.on_clicked(lambda evt: self.toggle_play())

        self.tb_start.on_submit(self._on_start_submit)
        self.tb_end.on_submit(self._on_end_submit)
        self.output_filename = "output_data.json"
        self.tb_output_name.on_submit(self._on_output_name_submit)
        
        # NEW WIRING for the keypoint input (updates visuals on Enter)
        self.tb_new_points.on_submit(self._on_new_points_submit)
        
        # NEW WIRING for the SAVE button (collects and saves data)
        self.btn_save_points.on_clicked(self._on_save_data_handler) # Renamed to new function

        self.slider.on_changed(self._on_slider)
        self.fps_slider.on_changed(self._on_fps_changed)

    def _draw_frame(self, i):
        x, y, z = self._get_xyz(i)
        frame_label = self.keys[i]
        
        if x is None:
            # Hide if missing data on this frame
            self.scat._offsets3d = ([], [], [])
            for ln, a, b in self.lines:
                ln.set_data_3d([], [], [])
            self.custom_line.set_data_3d([],[],[])
            self.custom_line_label.set_text("")
            self.fig.canvas.draw_idle()
            return

        # 1. Update scatter plot
        self.scat._offsets3d = (x, y, z)

        # 2. Update standard lines (SMPL edges)
        n = len(x)
        for ln, a, b in self.lines:
            if a < n and b < n:
                ln.set_data_3d([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
            else:
                ln.set_data_3d([], [], [])

        # 3. Update custom line and label
        if self.custom_line_points:
            a,b = self.custom_line_points
            if a < len(x) and b < len(x):
                self.custom_line.set_data_3d([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
                
                # Calculate distance for label
                kp_distance = distTwoPoints(x[a], y[a], z[a], x[b], y[b], z[b])
                real_life_distance = kp_distance * self.sf_vertical
                mid_x = (x[a] + x[b]) / 2
                mid_y = (y[a] + y[b]) / 2
                mid_z = (z[a] + z[b]) / 2
                
                self.custom_line_label.set_position((mid_x, mid_y, mid_z))
                self.custom_line_label.set_text(f"{real_life_distance:.2f} in.")
            else:
                self.custom_line.set_data_3d([], [], [])
                self.custom_line_label.set_text("")

        # 4. Update limits if not fixed
        if self.fixed_limits is None:
            self._set_limits(x, y, z)

        # 5. Update title and redraw
        who = f"idx={self.target_idx}" if self.target_idx is not None else "first person"
        self.ax.set_title(f"3D Pose Player — {frame_label} ({who})")
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # 2.4. EVENT HANDLERS
    # ---------------------------------------------------------------------

    def _on_key(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key == "left":
            self.step(-1)
        elif event.key == "right":
            self.step(1)
        elif event.key == "r":
            self.ax.view_init(elev=90, azim=90)
            self.fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(self.fig)

    def _on_slider(self, val):
        self.i = int(val)
        self._draw_frame(self.i)

    def _on_fps_changed(self, val):
        self.fps = int(val)
        self.interval = int(1000 / self.fps)
        self.timer.interval = self.interval        

    def _on_timer(self):
        if not self.playing:
            return
        self.i = (self.i + 1) % len(self.keys)
        self.slider.set_val(self.i)  # also triggers _draw_frame

    def _clamp_idx(self, v: int) -> int:
        return max(0, min(len(self.keys) - 1, v))

    def _on_start_submit(self, text):
        try:
            v = int(float(text))
        except ValueError:
            v = self.selected_start
        v = self._clamp_idx(v)
        # ensure start <= end
        if v > self.selected_end:
            self.selected_end = v
            self.tb_end.set_val(str(self.selected_end))
        self.selected_start = v
        self.tb_start.set_val(str(self.selected_start))

    def _on_end_submit(self, text):
        try:
            v = int(float(text))
        except ValueError:
            v = self.selected_end
        v = self._clamp_idx(v)
        # ensure start <= end
        if v < self.selected_start:
            self.selected_start = v
            self.tb_start.set_val(str(self.selected_start))
        self.selected_end = v
        self.tb_end.set_val(str(self.selected_end))

    # --- NEW HANDLERS FOR TEXTBOX INPUT ---

    def _on_new_points_submit(self, text):
        """
        Updates the custom line points based on the textbox input, and redraws.
        Wired to self.tb_new_points (on 'Enter' press).
        """
        try:
            a, b = map(int, text.split(","))
            if a >= 0 and b >= 0:
                # Use the existing method to update state and visual line
                self.connect_points(a, b) 
                print(f"[Tracking Updated] New points: {a} and {b}")
            else:
                raise ValueError("Indices must be non-negative.")
        except ValueError as e:
            print(f"Invalid keypoint format: {e}. Please enter two comma-separated positive integers (e.g., 10,3).")
            # Revert the textbox text if input was bad
            start_kp, end_kp = self.custom_line_points if self.custom_line_points else (0, 0)
            self.tb_new_points.set_val(f"{start_kp},{end_kp}")

    def _on_output_name_submit(self, text):
        """
        Updates the output filename when the user submits the textbox content.
        """
        if text.strip():
            self.output_filename = text.strip()
            print(f"[Filename] Output file set to: {self.output_filename}")
        else:
            # Revert to default if user cleared the box
            self.output_filename = "output_data.json"
            self.tb_output_name.set_val(self.output_filename)
            print(f"[Filename] Invalid name. Reverted to: {self.output_filename}")



    def _on_save_data_handler(self, _evt):
        """
        Handles the 'Save Points' button click.
        1. Ensures the latest points from the TextBox are loaded.
        2. Collects distance data for those points across the selected range.
        3. Saves the data.
        Wired to self.btn_save_points.
        """
        # 1. Ensure the internal state matches the TextBox content
        # This calls _on_new_points_submit to parse and set the points
        self._on_new_points_submit(self.tb_new_points.text)
        
        # 2. Check if valid points exist
        if not self.custom_line_points:
            print("Cannot save: No valid custom keypoints are set.")
            return

        # 3. Core Save Logic (replaces the old _on_mark_range content)
        print(f"[FrameRange] start={self.selected_start}, end={self.selected_end}")
        
        # filecleanupsingle(...) is commented out as before
        
        self._collect_segment_distances()
        createSegmentJson(self.collected_data, self.custom_line_points, self.output_filename)


    # The old _on_mark_range and _on_save_custom_distance are REMOVED 
    # as their functionality is replaced by the Matplotlib TextBox/Button logic.


    # ---------------------------------------------------------------------
    # 2.5. PUBLIC CONTROLS AND RUN METHOD
    # ---------------------------------------------------------------------

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def step(self, delta):
        self.playing = False
        self.btn_play.label.set_text("Play")
        self.timer.stop()
        self.i = (self.i + delta) % len(self.keys)
        self.slider.set_val(self.i)  # updates plot

    def get_frame_range(self):
        # Optional convenience getter if you want to read them from code
        return self.selected_start, self.selected_end

    def run(self):
        self._draw_frame(self.i)
        plt.show()