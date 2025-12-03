import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider

# =========================================================================
# 1. CONSTANTS AND DATA LOADING HELPERS (MODIFIED)
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
    Loads 3D keypoint data from a JSON file.
    
    MODIFIED: This now extracts the *first person's* keypoints data
    from the 'detections' list in each frame object.
    The resulting list will hold the object containing the 'keypoints_3d_m'.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                 raise TypeError("JSON content must be a list of frame objects.")
                 
            # Extract the relevant data block for the first person in each frame
            processed_frames = []
            for frame in data:
                detections = frame.get("detections")
                if detections and isinstance(detections, list) and len(detections) > 0:
                    # Assuming we are interested in the first detected person (index 0)
                    keypoint_data = detections[0]
                    # Also transfer the original frame index for the title
                    keypoint_data["pose_index"] = frame.get("frame_index")
                    processed_frames.append(keypoint_data)
                    
            return processed_frames
            
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")

def get_xyz_from_frame(frame_data):
    """
    Extracts (x, y, z) arrays from a single frame dictionary.
    
    MODIFIED: Keypoint structure is now 'keypoints_3d_m' and we slice out 
    the 4th confidence value.
    """
    # Key name changed from 'keypoints_3d_mm' to 'keypoints_3d_m' based on user's sample
    kp = frame_data.get("keypoints_3d_m") 
    
    if isinstance(kp, list) and kp and isinstance(kp[0], list) and len(kp[0]) >= 4:
        kp_array = np.array(kp, dtype=float)
        
        # Take the first 3 columns (x, y, z), ignoring the 4th (confidence)
        # Note: The order in your example is X, Y, Z.
        return kp_array[:, 0], kp_array[:, 1], kp_array[:, 2]
    
    return None, None, None


# =========================================================================
# 2. MAIN PLAYER CLASS (Refactored for Fixed Axis Limits on First Frame)
# =========================================================================

class ListPose3DPlayer:
    def __init__(
        self,
        json_path,
        edges=SKELETON_EDGES,
        fps=15,
        point_size=50
    ):
        # --- Config & Data Loading ---
        self.frames = load_frames(json_path) # frames is now a list
        self.num_frames = len(self.frames)
        if self.num_frames == 0:
            raise RuntimeError("No frames found or no detections with keypoints found in JSON.")
            
        self.fps = max(1, int(fps))
        self.edges = edges
        self.interval = int(1000 / self.fps)
        self.point_size = point_size
        
        # --- Scaling Factor (NEW) ---
        # Data is in meters, so we convert to millimeters (or arbitrary larger unit) for better plot range/scaling
        self.scale_factor = 1000.0 
        
        # --- State Variables ---
        self.i = 0 # Current index in the self.frames list
        self.playing = False
        
        # --- Axis Limits Storage ---
        self.fixed_x_lim = None
        self.fixed_y_lim = None
        self.fixed_z_lim = None
        
        # --- Figure & Axes Setup ---
        self.fig = plt.figure(figsize=(10, 8))
        # Adjust subplot for widgets at the bottom
        plt.subplots_adjust(bottom=0.20) 
        self.ax = self.fig.add_subplot(111, projection="3d")

        # --- Initial Frame Data & Artists ---
        x, y, z = self._get_xyz(self.i)
        
        # Apply scaling to the initial data (NEW)
        if x is not None:
            x, y, z = x * self.scale_factor, y * self.scale_factor, z * self.scale_factor
        
        if x is None:
            raise RuntimeError("Could not find 3D keypoint data in the first frame.")

        # --- Axes Limits Setup (REVISED) ---
        # Calculate and fix the limits based *ONLY* on the first frame (x, y, z)
        # It now uses a fixed range of 5.0 units.
        self._set_limits(x, y, z, initial_setup=True) 

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
        """Retrieves coordinates using the list index."""
        frame_data = self.frames[idx]
        return get_xyz_from_frame(frame_data)

    def _set_limits(self, x, y, z, margin_factor=1.05, initial_setup=False):
        """
        Sets cubic limits based on the current data range *only* if initial_setup is True.
        When initial_setup is True, it uses a fixed range of 5.0 centered on the pose. (MODIFIED)
        """
        if initial_setup:
            xs = np.array(x); ys = np.array(y); zs = np.array(z)
            
            # --- CALCULATE CENTERS (Based on first frame data) ---
            center_x = (xs.max() + xs.min()) / 2.0
            center_y = (ys.max() + ys.min()) / 2.0
            center_z = (zs.max() + zs.min()) / 2.0
            
            # --- USE FIXED RANGE OF 5.0 AS REQUESTED (MODIFIED: changed from 5.0 to 5000.0 because of scale factor) ---
            # Use a fixed range of 5.0 meters, which is 5000 units after scaling.
            FIXED_RANGE = 5000.0 
            max_range = FIXED_RANGE # Use 5000.0 as the total span
            
            # Calculate limit from max_range and margin
            limit = (max_range / 2.0) * margin_factor

            # Store the limits for future frames
            self.fixed_x_lim = (center_x - limit, center_x + limit)
            self.fixed_y_lim = (center_y - limit, center_y + limit)
            self.fixed_z_lim = (center_z - limit, center_z + limit)

            # Set the limits on the axes
            self.ax.set_xlim(*self.fixed_x_lim)
            self.ax.set_ylim(*self.fixed_y_lim)
            self.ax.set_zlim(*self.fixed_z_lim)
            
            # Labels from your threeDimPlot.py
            self.ax.set_xlabel("Left-Right X") # Updated based on common 3D convention/your example data
            self.ax.set_ylabel("Up-Down Y")    # Updated
            self.ax.set_zlabel("Forward-Back Z") # Updated
        
        # If not initial setup, do nothing to the limits (they remain fixed)
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
        x, y, z = self._get_xyz(i)
        
        # Get the frame number from the current frame object
        frame_number = self.frames[i].get("pose_index", i)
        
        # Apply scaling to the current frame data (NEW)
        if x is not None:
            x, y, z = x * self.scale_factor, y * self.scale_factor, z * self.scale_factor
        
        if x is None:
            # Hide if missing data on this frame
            self.scat._offsets3d = ([], [], [])
            for ln, a, b in self.lines:
                ln.set_data_3d([], [], [])
            self.fig.canvas.draw_idle()
            return

        # 1. Update scatter plot
        self.scat._offsets3d = (x, y, z)

        # 2. Update standard lines (Edges)
        n = len(x)
        for ln, a, b in self.lines:
            if a < n and b < n:
                ln.set_data_3d([x[a], x[b]], [y[a], y[b]], [z[a], z[b]])
            else:
                ln.set_data_3d([], [], [])

        # 3. Update title and redraw
        self.ax.set_title(f"3D Pose Player — Pose Index {frame_number} (List Index {i})")
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
    # --- IMPORTANT: REPLACE WITH YOUR ACTUAL JSON FILE PATH ---
    JSON_FILE_PATH = '/Users/franciscojimenez/Desktop/DataCleaning-MartialArtsProject-/realsense_output/3d_pose_reconstruction.json'
    
    player = ListPose3DPlayer(
        json_path=JSON_FILE_PATH,
        edges=SKELETON_EDGES,
        fps=20
    )
    player.run()