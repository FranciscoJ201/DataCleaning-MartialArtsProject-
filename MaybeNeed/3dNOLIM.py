import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider

# =========================================================================
# 1. CONSTANTS AND DATA LOADING HELPERS
# =========================================================================

# The custom skeleton edges you provided in your original script
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
]

def load_frames(json_path):
    """
    Loads 3D keypoint data from a JSON file which is a list of frame objects.
    Returns the list of frame data (self.frames).
    The length of this list is the number of frames.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                 raise TypeError("JSON content must be a list of frame objects.")
            return data
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")

def get_xyz_from_frame(frame_data):
    """
    Extracts (x, y, z) arrays from a single frame dictionary.
    """
    kp = frame_data.get("keypoints_3d_mm")
    
    if isinstance(kp, list) and kp and isinstance(kp[0], list) and len(kp[0]) >= 3:
        kp_array = np.array(kp, dtype=float)
        
        # Take the first 3 columns (x, y, z)
        return kp_array[:, 0], kp_array[:, 1], kp_array[:, 2]
    
    return None, None, None


# =========================================================================
# 2. MAIN PLAYER CLASS (Modified for Width Multiplier)
# =========================================================================

class ListPose3DPlayer:
    def __init__(
        self,
        json_path,
        edges=SKELETON_EDGES,
        fps=15,
        point_size=50,
        width_multiplier=1.0 # NEW PARAMETER for Y-axis scaling
    ):
        # --- Config & Data Loading ---
        self.frames = load_frames(json_path)
        self.num_frames = len(self.frames)
        if self.num_frames == 0:
            raise RuntimeError("No frames found in JSON list.")
            
        self.fps = max(1, int(fps))
        self.edges = edges
        self.interval = int(1000 / self.fps)
        self.point_size = point_size
        self.width_multiplier = width_multiplier # STORED ATTRIBUTE
        
        # --- State Variables ---
        self.i = 0
        self.playing = False
        
        # --- Figure & Axes Setup ---
        self.fig = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.20) 
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Set initial labels
        self.ax.set_xlabel("Left-Right Y (SCALED)") # Label updated for clarity
        self.ax.set_ylabel("Up-Down Z")
        self.ax.set_zlabel("Forward-Back X")

        # --- Initial Frame Data & Artists ---
        x, y, z = self._get_xyz(self.i) # This call now returns SCALED Y
        if x is None:
            raise RuntimeError("Could not find 3D keypoint data in the first frame.")

        # Limits will be calculated and set on the first draw and updated on every draw
        self._set_limits(x, y, z) 

        # Main keypoint scatter plot
        self.scat = self.ax.scatter(x, y, z, c='blue', marker='o', s=self.point_size)
        
        # Standard connections (lines for edges)
        self.lines = []
        for (a, b) in self.edges:
            if a < len(x) and b < len(x):
                ln, = self.ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]], c='red', linewidth=2)
                self.lines.append((ln, a, b))
        
        # --- UI Setup (Widgets) ---
        self._add_widgets()
        self.ax.view_init(elev=10, azim=-60)
        
        # --- Event Bindings ---
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._on_timer)


    # ---------------------------------------------------------------------
    # 2.1. DATA ACCESS AND LIMITS
    # ---------------------------------------------------------------------

    def _get_xyz(self, idx):
        """Retrieves coordinates using the list index and applies width scaling."""
        frame_data = self.frames[idx]
        x_raw, y_raw, z_raw = get_xyz_from_frame(frame_data)
        
        if x_raw is not None:
            # Apply the multiplier to the Y-coordinate (Left-Right axis)
            y_scaled = y_raw * self.width_multiplier
            return x_raw, y_scaled, z_raw
            
        return None, None, None


    def _set_limits(self, x, y, z, margin_factor=1.05):
        """
        Dynamically calculates and sets cubic limits based on the current data range.
        The plot will zoom/pan to keep the pose centered and fully visible.
        """
        if x is None or len(x) == 0:
            return

        xs = np.array(x); ys = np.array(y); zs = np.array(z)

        # 1. Calculate the range (max_val - min_val) across all axes for the current frame
        range_x = xs.max() - xs.min()
        range_y = ys.max() - ys.min()
        range_z = zs.max() - zs.min()
        
        # 2. Find the largest range to ensure the plot is cubic (equal aspect ratio)
        max_range = max(range_x, range_y, range_z)
        max_range = max(max_range, 0.05) 
        
        # 3. Calculate the half-margin to apply for padding
        half_margin = (max_range * margin_factor) / 2.0
        
        # 4. Calculate the center of the total 3D data envelope
        center_x = (xs.max() + xs.min()) / 2.0
        center_y = (ys.max() + ys.min()) / 2.0
        center_z = (zs.max() + zs.min()) / 2.0

        # 5. Apply the limits
        self.ax.set_xlim(center_x - half_margin, center_x + half_margin)
        self.ax.set_ylim(center_y - half_margin, center_y + half_margin)
        self.ax.set_zlim(center_z - half_margin, center_z + half_margin)


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
        x, y, z = self._get_xyz(i)
        
        # Get the frame number from the current frame object
        frame_number = self.frames[i].get("pose_index", i)
        
        if x is None:
            # Hide if missing data on this frame
            self.scat._offsets3d = ([], [], [])
            for ln, a, b in self.lines:
                ln.set_data_3d([], [], [])
            self.ax.set_title(f"3D Pose Player — Pose Index {frame_number} (List Index {i}) - NO DATA")
            self.fig.canvas.draw_idle()
            return

        # 1. Update limits (Crucial change: always update limits)
        self._set_limits(x, y, z)

        # 2. Update scatter plot
        self.scat._offsets3d = (x, y, z)

        # 3. Update standard lines (Edges)
        n = len(x)
        for ln, a, b in self.lines:
            if a < n and b < n:
                ln.set_data_3d([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
            else:
                ln.set_data_3d([], [], [])

        # 4. Update title and redraw
        self.ax.set_title(f"3D Pose Player — Pose Index {frame_number} (List Index {i}) | Width Multiplier: {self.width_multiplier:.1f}")
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

# =========================================================================
# 3. EXAMPLE USAGE
# =========================================================================

if __name__ == '__main__':
    # --- IMPORTANT: REPLACE WITH YOUR ACTUAL JSON FILE PATH ---
    JSON_FILE_PATH = '/Users/franciscojimenez/Desktop/DataCleaning-MartialArtsProject-/3d_pose_reconstruction.json'
    
    # *** Set width_multiplier here. 1.0 is no change, 2.0 makes it double-width. ***
    MULTIPLIER = 5.0
    
    player = ListPose3DPlayer(
        json_path=JSON_FILE_PATH,
        edges=SKELETON_EDGES,
        fps=20,
        width_multiplier=MULTIPLIER
    )
    player.run()