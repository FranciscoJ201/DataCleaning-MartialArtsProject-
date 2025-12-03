import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider

# =========================================================================
# 1. CONSTANTS AND DATA LOADING HELPERS (REVISED)
# =========================================================================

# The custom skeleton edges 
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
]

# Distinct colors for up to two people
PERSON_COLORS = ['blue', 'red'] 
MAX_PEOPLE = 2 # Plotting limit

def load_frames(json_path):
    """
    Loads ALL frame data, including the full list of detections for each frame.
    
    REVISED: Returns the raw list of frame objects.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                 raise TypeError("JSON content must be a list of frame objects.")
            return data
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")

def get_person_keypoints(frame_data, person_index):
    """
    Extracts (x, y, z) arrays from a single person's detection in a frame.
    
    If the person is not found, returns None, None, None.
    """
    detections = frame_data.get("detections")
    
    if (detections and 
        isinstance(detections, list) and 
        len(detections) > person_index):
        
        person_data = detections[person_index]
        kp = person_data.get("keypoints_3d_m")
        
        if isinstance(kp, list) and kp and isinstance(kp[0], list) and len(kp[0]) >= 4:
            # np.array converts [None, None, None, conf] to [nan, nan, nan, conf]
            kp_array = np.array(kp, dtype=float) 
            
            # Take the first 3 columns (x, y, z)
            return kp_array[:, 0], kp_array[:, 1], kp_array[:, 2]
    
    return None, None, None


# =========================================================================
# 2. MULTI-POSE PLAYER CLASS (REVISED)
# =========================================================================

class MultiPose3DPlayer:
    def __init__(
        self,
        json_path,
        edges=SKELETON_EDGES,
        fps=15,
        point_size=50
    ):
        # --- Config & Data Loading ---
        # self.frames now holds the full list of frame objects
        self.frames = load_frames(json_path) 
        self.num_frames = len(self.frames)
        if self.num_frames == 0:
            raise RuntimeError("No frames found in JSON list.")
            
        self.fps = max(1, int(fps))
        self.edges = edges
        self.interval = int(1000 / self.fps)
        self.point_size = point_size
        self.scale_factor = 1000.0 # Convert meters to plot units
        self.max_people = MAX_PEOPLE
        
        # --- State Variables ---
        self.i = 0 # Current index
        self.playing = False
        
        # --- Axis Limits Storage ---
        self.fixed_x_lim = None
        self.fixed_y_lim = None
        self.fixed_z_lim = None
        
        # --- Figure & Axes Setup ---
        self.fig = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.20) 
        self.ax = self.fig.add_subplot(111, projection="3d")

        # --- Initial Frame Data & Axes Limits Setup ---
        # We need the combined data of all detected people in frame 0 for initial limits
        all_x, all_y, all_z = [], [], []
        
        for p_id in range(self.max_people):
            x, y, z = get_person_keypoints(self.frames[0], p_id)
            if x is not None:
                all_x.extend(x); all_y.extend(y); all_z.extend(z)
        
        if not all_x:
            raise RuntimeError("Could not find any 3D keypoint data in the first frame.")
            
        # Apply scaling to the combined data
        all_x, all_y, all_z = (np.array(all_x) * self.scale_factor, 
                               np.array(all_y) * self.scale_factor, 
                               np.array(all_z) * self.scale_factor)

        # Calculate and fix limits based on all people in frame 0 (ignoring NaNs)
        self._set_limits(all_x, all_y, all_z, initial_setup=True) 

        # --- Initialize Multi-Artists (NEW) ---
        self.all_scats = []
        self.all_lines = []

        for p_id in range(self.max_people):
            color = PERSON_COLORS[p_id % len(PERSON_COLORS)]
            x, y, z = get_person_keypoints(self.frames[0], p_id)
            
            if x is not None:
                x, y, z = x * self.scale_factor, y * self.scale_factor, z * self.scale_factor
                
                # Filter NaNs for initial plotting
                valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
                
                # 1. Scatter Artist
                scat = self.ax.scatter(x_valid, y_valid, z_valid, c=color, marker='o', s=self.point_size)
                self.all_scats.append(scat)

                # 2. Line Artists (Skeleton)
                person_lines = []
                for (a, b) in self.edges:
                    ln, = self.ax.plot([], [], [], c=color, linewidth=2)
                    person_lines.append((ln, a, b))
                self.all_lines.append(person_lines)
            else:
                # Append None or dummy artists if fewer than MAX_PEOPLE exist initially
                dummy_scat = self.ax.scatter([], [], [], c=color, marker='o', s=self.point_size)
                self.all_scats.append(dummy_scat)

                dummy_lines = []
                for (a, b) in self.edges:
                    ln, = self.ax.plot([], [], [], c=color, linewidth=2)
                    dummy_lines.append((ln, a, b))
                self.all_lines.append(dummy_lines)
                
        # --- UI Setup (Widgets) ---
        self._add_widgets()
        self.ax.view_init(elev=10, azim=-60)
        
        # --- Event Bindings ---
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._on_timer)


    # ---------------------------------------------------------------------
    # 2.1. DATA ACCESS AND LIMITS
    # ---------------------------------------------------------------------

    def _set_limits(self, x, y, z, margin_factor=1.05, initial_setup=False):
        """
        Sets cubic limits based on the first frame data, ignoring NaNs. (REUSED)
        """
        if initial_setup:
            xs = np.array(x); ys = np.array(y); zs = np.array(z)
            
            # CALCULATE MIN/MAX/CENTER using nanmax/nanmin (safely ignoring NaNs)
            x_min, x_max = np.nanmin(xs), np.nanmax(xs)
            y_min, y_max = np.nanmin(ys), np.nanmax(ys)
            z_min, z_max = np.nanmin(zs), np.nanmax(zs)
            
            if np.isnan(x_min) or np.isnan(x_max):
                 # Fallback if ALL keypoints of ALL people are missing in the first frame
                 raise RuntimeError("All keypoints in the first frame are invalid (NaN). Cannot set plot limits.")

            center_x = (x_max + x_min) / 2.0
            center_y = (y_max + y_min) / 2.0
            center_z = (z_max + z_min) / 2.0
            
            # USE FIXED RANGE (5000 units = 5.0 meters)
            FIXED_RANGE = 5000.0 
            max_range = FIXED_RANGE 
            limit = (max_range / 2.0) * margin_factor

            self.fixed_x_lim = (center_x - limit, center_x + limit)
            self.fixed_y_lim = (center_y - limit, center_y + limit)
            self.fixed_z_lim = (center_z - limit, center_z + limit)

            self.ax.set_xlim(*self.fixed_x_lim)
            self.ax.set_ylim(*self.fixed_y_lim)
            self.ax.set_zlim(*self.fixed_z_lim)
            
            self.ax.set_xlabel("Left-Right X")
            self.ax.set_ylabel("Up-Down Y")
            self.ax.set_zlabel("Forward-Back Z")
        
        pass 


    # ---------------------------------------------------------------------
    # 2.2. UI AND DRAWING METHODS
    # ---------------------------------------------------------------------

    def _add_widgets(self):
        # Axes for the controls
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        ax_prev   = plt.axes([0.15, 0.1, 0.1, 0.05])
        ax_play   = plt.axes([0.27, 0.1, 0.1, 0.05])
        ax_next   = plt.axes([0.39, 0.1, 0.1, 0.05])

        # Slider and Buttons
        self.slider = Slider(ax_slider, "Frame", 0, self.num_frames - 1, valinit=self.i, valstep=1)
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, "Next")

        # Wire up controls
        self.btn_prev.on_clicked(lambda evt: self.step(-1))
        self.btn_next.on_clicked(lambda evt: self.step(1))
        self.btn_play.on_clicked(lambda evt: self.toggle_play())
        self.slider.on_changed(self._on_slider)

    def _draw_frame(self, i):
        frame_data = self.frames[i]
        frame_number = frame_data.get("frame_index", i)
        
        # --- Update all MAX_PEOPLE artists (NEW DRAWING LOOP) ---
        for p_id in range(self.max_people):
            x, y, z = get_person_keypoints(frame_data, p_id)
            
            scat = self.all_scats[p_id]
            person_lines = self.all_lines[p_id]

            if x is not None:
                # Apply scaling and convert to numpy array
                x, y, z = np.array(x) * self.scale_factor, np.array(y) * self.scale_factor, np.array(z) * self.scale_factor
                
                # Filter NaNs for scatter plot
                valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
                
                # 1. Update scatter plot (only drawing valid points)
                scat._offsets3d = (x_valid, y_valid, z_valid)

                # 2. Update lines (Edges)
                n = len(x)
                for ln, a, b in person_lines:
                    # Check array bounds AND check if both points are valid (not NaN)
                    if (a < n and b < n and 
                        np.isfinite(x[a]) and np.isfinite(x[b]) and 
                        np.isfinite(y[a]) and np.isfinite(y[b]) and 
                        np.isfinite(z[a]) and np.isfinite(z[b])):
                        
                        ln.set_data_3d([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
                    else:
                        # Hide the line segment if either endpoint is missing/NaN
                        ln.set_data_3d([], [], [])
            else:
                # Person not detected in this frame, hide their artists
                scat._offsets3d = ([], [], [])
                for ln, a, b in person_lines:
                    ln.set_data_3d([], [], [])

        # 3. Update title and redraw
        self.ax.set_title(f"3D Multi-Pose Player — Frame Index {frame_number} (List Index {i})")
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # 2.3. EVENT HANDLERS AND PUBLIC CONTROLS
    # ---------------------------------------------------------------------

    def _on_slider(self, val):
        self.i = int(val)
        self._draw_frame(self.i)

    def _on_timer(self):
        if not self.playing:
            return
        # Loop through frames
        self.i = (self.i + 1) % self.num_frames
        self.slider.set_val(self.i) # updates plot via _on_slider

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
        self.i = (self.i + delta) % self.num_frames
        self.slider.set_val(self.i) # updates plot

    def run(self):
        self._draw_frame(self.i)
        plt.show()


if __name__ == '__main__':
    JSON_FILE_PATH = '/Users/franciscojimenez/Desktop/DataCleaning-MartialArtsProject-/realsense_output/3d_pose_reconstruction.json'
    
    player = MultiPose3DPlayer(
        json_path=JSON_FILE_PATH,
        edges=SKELETON_EDGES,
        fps=20
    )
    player.run()