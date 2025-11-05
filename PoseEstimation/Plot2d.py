import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from collections import defaultdict

def plot(input):
    # --- 1. CONFIGURATION & DATA STRUCTURE ---

    # Standard YOLOv8 Keypoint Indices (0-16)
    # Keypoint 0 is the nose, used for labeling the ID.
    SKELETON_EDGES = [
        (15, 13), (13, 11), (16, 14), (14, 12),  # Legs
        (11, 12), (5, 11), (6, 12),              # Hips/Torso connection
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), # Shoulders/Arms
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)   # Face/Head
    ]

    # Path to the JSON file created by the RE-ID script
    JSON_FILE_PATH = input

    # Color cycle for drawing different people
    ID_COLORS = plt.cm.get_cmap('hsv', 10) 


    def load_pose_data(file_path):
        """
        Loads all tracked data and groups it by frame index.
        Returns: {frame_idx: [detection_data_person_1, detection_data_person_2, ...]}
        """
        try:
            with open(file_path, 'r') as f:
                all_detections = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at '{file_path}'. Please run the pose extractor and re-ID scripts first.")
            return None, 0, 0

        # Group detections by frame index
        frames_map = defaultdict(list)
        max_x, max_y = 0, 0

        for d in all_detections:
            frame_idx = d['frame_index']
            keypoints_list = d['keypoints_xyz']
            
            # Convert keypoints to a numpy array (x, y, conf)
            keypoints_array = np.array(keypoints_list, dtype=float)
            
            # Store keypoints and ID for the frame
            frames_map[frame_idx].append({
                'keypoints': keypoints_array,
                'track_id': d.get('track_id', -1)
            })
            
            # Update plot limits
            if keypoints_array.size > 0:
                max_x = max(max_x, np.max(keypoints_array[:, 0]))
                max_y = max(max_y, np.max(keypoints_array[:, 1]))

        # Get a sorted list of frames
        sorted_frame_indices = sorted(frames_map.keys())
        
        return {idx: frames_map[idx] for idx in sorted_frame_indices}, int(max_x * 1.05), int(max_y * 1.05)


    # --- 2. INTERACTIVE VIEWER CLASS ---

    class PoseViewer:
        
        def __init__(self, frames_data, x_limit, y_limit):
            if not frames_data:
                raise ValueError("No pose data available to visualize.")

            # --- State ---
            self.data = frames_data
            self.sorted_frames = sorted(frames_data.keys())
            self.num_frames = len(self.sorted_frames)
            self.current_frame_idx = 0
            self.is_playing = False
            self.fps = 15
            self.interval = int(1000 / self.fps)

            # --- Figure and Axes Setup (2D Plot) ---
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.subplots_adjust(bottom=0.25) # Make room for controls
            
            # Set boundaries and style
            self.ax.set_ylim(y_limit, 0) # Invert Y-axis to match image coordinates (origin top-left)
            self.ax.set_xlim(0, x_limit)
            self.ax.set_xlabel('X Position (Pixels)')
            self.ax.set_ylabel('Y Position (Pixels)')
            self.ax.set_aspect('equal', adjustable='box') 
            self.ax.grid(True)
            
            # Initialize Artists: Use lists to hold artists for multiple people
            self.scatters = []
            self.lines = []
            self.labels = []

            # --- Widget Setup ---
            self._setup_widgets()
            
            # --- Timer Setup ---
            self.timer = self.fig.canvas.new_timer(interval=self.interval)
            self.timer.add_callback(self._on_timer)
            
            # Initial draw
            self._draw_frame(0)


        def _setup_widgets(self):
            # Axes for the slider (bottom)
            ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
            self.slider = Slider(
                ax_slider, 'Frame', 0, self.num_frames - 1, 
                valinit=0, valstep=1, valfmt='%0.0f'
            )
            self.slider.on_changed(self._on_slider_change)

            # Axes for the play/pause button (below the slider)
            ax_button = plt.axes([0.45, 0.04, 0.1, 0.07])
            self.button = Button(ax_button, 'Play', color='lightblue', hovercolor='0.975')
            self.button.on_clicked(self.toggle_play)

        
        def _draw_frame(self, frame_idx_in_list):
            """Updates the plot data for a given frame index in the sorted list."""
            
            if frame_idx_in_list < 0 or frame_idx_in_list >= self.num_frames:
                return

            # Get the actual frame number from the sorted list
            actual_frame_number = self.sorted_frames[frame_idx_in_list]
            people_in_frame = self.data.get(actual_frame_number, [])
            num_people = len(people_in_frame)
            
            # Clear previous artists if necessary
            for artist in self.scatters + self.lines + self.labels:
                artist.remove()
            self.scatters = []
            self.lines = []
            self.labels = []

            # Draw each person
            for i, person in enumerate(people_in_frame):
                kp = person['keypoints']
                track_id = person['track_id']
                color = ID_COLORS(track_id % 10) # Assign color based on ID
                
                x_points = kp[:, 0]
                y_points = kp[:, 1]
                
                # 1. Draw Keypoints (Scatter Plot)
                scatter = self.ax.scatter(x_points, y_points, c=[color], s=50, marker='o', zorder=2)
                self.scatters.append(scatter)
                
                # 2. Draw Lines (Skeleton)
                for a, b in SKELETON_EDGES:
                    x_data = [kp[a, 0], kp[b, 0]]
                    y_data = [kp[a, 1], kp[b, 1]]
                    line, = self.ax.plot(x_data, y_data, c=color, linewidth=2, alpha=0.7, zorder=1)
                    self.lines.append(line)
                
                # 3. Draw Track ID Label (near the nose, keypoint 0)
                if kp.shape[0] > 0:
                    # Draw ID label slightly above the nose (kp[0])
                    label_x = kp[0, 0]
                    label_y = kp[0, 1] - 15  # Adjust label position slightly above the nose
                    label = self.ax.text(label_x, label_y, f'ID: {track_id}', 
                                        color=color, fontsize=10, 
                                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
                    self.labels.append(label)

            # Update Title
            self.ax.set_title(f"Pose Viewer | Frame: {actual_frame_number} / Max ID: {self._get_max_id()}")
            
            # Redraw the canvas
            self.fig.canvas.draw_idle()


        def _get_max_id(self):
            """Helper to find the highest track ID used across all frames."""
            max_id = 0
            for frame_data in self.data.values():
                for person in frame_data:
                    max_id = max(max_id, person['track_id'])
            return max_id
        
        
        def _on_timer(self):
            """The callback function for the animation timer."""
            if not self.is_playing:
                return
                
            # Move to the next frame (loop back to 0 if at the end)
            self.current_frame_idx = (self.current_frame_idx + 1) % self.num_frames
            self.slider.set_val(self.current_frame_idx)


        def toggle_play(self, event=None):
            """Toggles play/pause state."""
            self.is_playing = not self.is_playing
            
            if self.is_playing:
                self.button.label.set_text('Pause')
                self.timer.start()
            else:
                self.button.label.set_text('Play')
                self.timer.stop()
                
            if self.is_playing and self.current_frame_idx == self.num_frames - 1:
                self.current_frame_idx = 0
                self.slider.set_val(0)


        def _on_slider_change(self, val):
            """Handler for when the slider is moved."""
            frame_idx_in_list = int(val)
            self.current_frame_idx = frame_idx_in_list
            self._draw_frame(frame_idx_in_list)


        def run(self):
            """Displays the Matplotlib figure."""
            plt.show()


    # --- 3. MAIN EXECUTION ---

    
    frames_data, x_limit, y_limit = load_pose_data(JSON_FILE_PATH)

    if frames_data:
        try:
            viewer = PoseViewer(frames_data, x_limit, y_limit)
            print("\nInteractive Multi-Person Pose Viewer Ready: Click 'Play' or use the slider to scrub frames.")
            viewer.run()
        except Exception as e:
            print(f"An error occurred during visualization: {e}")

    