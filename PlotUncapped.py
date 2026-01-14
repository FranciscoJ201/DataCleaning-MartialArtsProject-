import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider, TextBox 
import math 

# =========================================================================
# 1. CONSTANTS, UTILS, AND DATA LOADING HELPERS 
# =========================================================================

def distTwoPoints(x1,y1,z1,x2,y2,z2):
    return np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )

# --- REFERENCE KEYPOINTS FOR VERTICAL SCALING ---
REF_HEAD_IDX = 0  # Nose
REF_FOOT_IDX = 15 # Left Ankle

SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
]
LIMB_SEGMENTS_TO_SAVE = [
        (14, 16, "Right_Knee_to_Ankle"), 
        (14, 12, "Right_Knee_to_Hip"),   
        (10, 8, "Right_Wrist_to_Elbow"), 
        (8, 6, "Right_Elbow_to_Shoulder"), 
        (13, 15, "Left_Knee_to_Ankle"),  
        (13, 11, "Left_Knee_to_Hip"),    
        (9, 7, "Left_Wrist_to_Elbow"),   
        (7, 5, "Left_Elbow_to_Shoulder"),  
    ]

KEYPOINT_NAMES = [
    "Nose (0)", 
    "Left Eye (1)", "Right Eye (2)", "Left Ear (3)", "Right Ear (4)",
    "Left Shoulder (5)", "Right Shoulder (6)", 
    "Left Elbow (7)", "Right Elbow (8)", 
    "Left Wrist (9)", "Right Wrist (10)",
    "Left Hip (11)", "Right Hip (12)",
    "Left Knee (13)", "Right Knee (14)",
    "Left Ankle (15)", "Right Ankle (16)" 
]

PERSON_COLORS = [
    'blue', 'red', 'green', 'orange', 'purple', 
    'cyan', 'magenta', 'yellow', 'black', 'brown'
]

CUSTOM_LINE_COLOR = 'magenta' 
TARGET_PERSON_IDX = 0 

def load_frames(json_path):
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                 raise TypeError("JSON content must be a list of frame objects.")
            return data
    except Exception as e:
        raise RuntimeError(f"Error loading JSON file: {e}")

def get_person_keypoints(frame_data, person_index):
    detections = frame_data.get("detections")
    if (detections and isinstance(detections, list) and len(detections) > person_index):
        person_data = detections[person_index]
        kp = person_data.get("keypoints_3d_m")
        if isinstance(kp, list) and kp and isinstance(kp[0], list) and len(kp[0]) >= 4:
            kp_array = np.array(kp, dtype=float) 
            return kp_array[:, 0], kp_array[:, 1], kp_array[:, 2]
    return None, None, None


# =========================================================================
# 2. MULTI-POSE PLAYER CLASS
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
        self.frames = load_frames(json_path) 
        self.num_frames = len(self.frames)
        self.LIMB_SEGMENTS_TO_SAVE = LIMB_SEGMENTS_TO_SAVE
        if self.num_frames == 0:
            raise RuntimeError("No frames found in JSON list.")
            
        self.fps = max(1, int(fps))
        self.edges = edges
        self.interval = int(1000 / self.fps)
        self.point_size = point_size
        self.scale_factor = 1000.0 
        
        # --- Dynamic Person Count ---
        max_detected = 0
        for f in self.frames:
            dets = f.get("detections", [])
            if isinstance(dets, list):
                max_detected = max(max_detected, len(dets))
        self.max_people = max_detected if max_detected > 0 else 1
        print(f"Loaded {self.num_frames} frames. Max people: {self.max_people}")

        # --- State Variables ---
        self.i = 0 
        self.playing = False
        self.custom_line_points = (0, 15) 
        self.keypoint_names = KEYPOINT_NAMES 
        self.sf_vertical = 1.0 
        self.ref_height_inches = 70.0 
        
        # --- Figure & Axes Setup ---
        self.fig = plt.figure(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.30) 
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        # --- Initial Frame Data & Axes Limits Setup ---
        all_x, all_y, all_z = [], [], []
        for p_id in range(self.max_people):
            x, y, z = get_person_keypoints(self.frames[0], p_id)
            if x is not None:
                x_arr, y_arr, z_arr = np.array(x), np.array(y), np.array(z)
                all_x.extend(x_arr); all_y.extend(y_arr); all_z.extend(z_arr)
        
        if not all_x:
            self._set_limits([], [], [], initial_setup=True, default=True)
        else:
            all_x, all_y, all_z = (np.array(all_x) * self.scale_factor, 
                                   np.array(all_y) * self.scale_factor, 
                                   np.array(all_z) * self.scale_factor)
            self._set_limits(all_x, all_y, all_z, initial_setup=True) 

        # --- DRAW CAMERA AT ORIGIN (0,0,0) ---
        self._draw_camera_visual()

        # --- Initialize Artists ---
        self.all_scats = []
        self.all_lines = []

        for p_id in range(self.max_people):
            color = PERSON_COLORS[p_id % len(PERSON_COLORS)]
            x, y, z = get_person_keypoints(self.frames[0], p_id)
            
            if x is not None:
                x_sc = np.array(x) * self.scale_factor
                y_sc = np.array(y) * self.scale_factor
                z_sc = np.array(z) * self.scale_factor
                
                valid_mask = np.isfinite(x_sc) & np.isfinite(y_sc) & np.isfinite(z_sc)
                x_valid, y_valid, z_valid = x_sc[valid_mask], y_sc[valid_mask], z_sc[valid_mask]
                
                scat = self.ax.scatter(x_valid, y_valid, z_valid, c=color, marker='o', s=self.point_size)
                self.all_scats.append(scat)

                person_lines = []
                for (a, b) in self.edges:
                    ln, = self.ax.plot([], [], [], c=color, linewidth=2)
                    person_lines.append((ln, a, b))
                self.all_lines.append(person_lines)
            else:
                dummy_scat = self.ax.scatter([], [], [], c=color, marker='o', s=self.point_size)
                self.all_scats.append(dummy_scat)

                dummy_lines = []
                for (a, b) in self.edges:
                    ln, = self.ax.plot([], [], [], c=color, linewidth=2)
                    dummy_lines.append((ln, a, b))
                self.all_lines.append(dummy_lines)
                
        self.custom_line, = self.ax.plot([],[],[], color=CUSTOM_LINE_COLOR, linewidth=3, linestyle='--')
        self.custom_line_label = self.ax.text(0,0,0,"", color=CUSTOM_LINE_COLOR,fontsize=12, fontweight='bold')
        
        # --- UI Setup ---
        self._add_widgets()
        
        # Set default view (Azimuth -90 means looking along Z axis approx)
        self.ax.view_init(elev=10, azim=-90)
        
        self.timer = self.fig.canvas.new_timer(interval=self.interval)
        self.timer.add_callback(self._on_timer)

    # ---------------------------------------------------------------------
    # NEW FUNCTION: DRAW CAMERA FRUSTUM
    # ---------------------------------------------------------------------
    def _draw_camera_visual(self):
        """Draws a wireframe pyramid at (0,0,0) to represent the RealSense camera."""
        # Define size (in mm, since data is scaled by 1000)
        size = 300 
        
        # Apex is at origin (camera position)
        apex = np.array([0, 0, 0])
        
        # Base of pyramid (image plane) pointing in +Z direction
        # RealSense Z is forward, X is right, Y is down
        z_depth = size
        w = size * 0.8
        h = size * 0.6
        
        # 4 corners
        tl = np.array([-w, -h, z_depth]) # Top-Left
        tr = np.array([ w, -h, z_depth]) # Top-Right
        br = np.array([ w,  h, z_depth]) # Bottom-Right
        bl = np.array([-w,  h, z_depth]) # Bottom-Left
        
        # Lines from apex
        color = 'black'
        self.ax.plot([apex[0], tl[0]], [apex[1], tl[1]], [apex[2], tl[2]], c=color, lw=2, linestyle='--')
        self.ax.plot([apex[0], tr[0]], [apex[1], tr[1]], [apex[2], tr[2]], c=color, lw=2, linestyle='--')
        self.ax.plot([apex[0], br[0]], [apex[1], br[1]], [apex[2], br[2]], c=color, lw=2, linestyle='--')
        self.ax.plot([apex[0], bl[0]], [apex[1], bl[1]], [apex[2], bl[2]], c=color, lw=2, linestyle='--')
        
        # Rectangle connecting corners
        self.ax.plot([tl[0], tr[0]], [tl[1], tr[1]], [tl[2], tr[2]], c=color, lw=2)
        self.ax.plot([tr[0], br[0]], [tr[1], br[1]], [tr[2], br[2]], c=color, lw=2)
        self.ax.plot([br[0], bl[0]], [br[1], bl[1]], [br[2], bl[2]], c=color, lw=2)
        self.ax.plot([bl[0], tl[0]], [bl[1], tl[1]], [bl[2], tl[2]], c=color, lw=2)

        # Label
        self.ax.text(0, -h*1.5, 0, "CAMERA\n(0,0,0)", color='black', ha='center', fontweight='bold')

    # ---------------------------------------------------------------------
    # 2.2. DRAWING AND UI IMPLEMENTATION
    # ---------------------------------------------------------------------

    def _set_limits(self, x, y, z, margin_factor=1.05, initial_setup=False, default=False):
        if initial_setup:
            if default:
                self.ax.set_xlim(-2000, 2000)
                self.ax.set_ylim(-2000, 2000)
                self.ax.set_zlim(0, 4000)
                return

            # Include the Camera Origin (0,0,0) in the limit calculation
            # This ensures the camera is not cut out of the view
            xs = np.append(np.array(x), 0)
            ys = np.append(np.array(y), 0)
            zs = np.append(np.array(z), 0)
            
            x_min, x_max = np.nanmin(xs), np.nanmax(xs)
            y_min, y_max = np.nanmin(ys), np.nanmax(ys)
            z_min, z_max = np.nanmin(zs), np.nanmax(zs)
            
            center_x = (x_max + x_min) / 2.0
            center_y = (y_max + y_min) / 2.0
            center_z = (z_max + z_min) / 2.0
            
            # Increase range slightly to ensure comfortable fit
            FIXED_RANGE = 5000.0 
            limit = (FIXED_RANGE / 2.0) * margin_factor

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


    def _add_widgets(self):
        ax_prev   = plt.axes([0.15, 0.20, 0.1, 0.05]) 
        ax_play   = plt.axes([0.27, 0.20, 0.1, 0.05])
        ax_next   = plt.axes([0.39, 0.20, 0.1, 0.05])
        ax_save   = plt.axes([0.15, 0.27, 0.12, 0.05])
        
        ax_kp_input = plt.axes([0.70, 0.20, 0.10, 0.05]) 
        ax_height_input = plt.axes([0.70, 0.12, 0.10, 0.05]) 
        
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])

        self.slider = Slider(ax_slider, "Frame", 0, self.num_frames - 1, valinit=self.i, valstep=1)
        self.btn_prev = Button(ax_prev, "Prev")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, "Next")
        self.btn_save = Button(ax_save, "Save Limb Lengths")

        start_kp, end_kp = self.custom_line_points
        self.tb_new_points = TextBox(
            ax_kp_input, 
            f"Person {TARGET_PERSON_IDX} Points (A,B)", 
            initial=f"{start_kp},{end_kp}"
        )
        
        self.tb_ref_height = TextBox(
            ax_height_input,
            "Height (Inches)",
            initial=f"{self.ref_height_inches:.1f}"
        )

        self.btn_prev.on_clicked(lambda evt: self.step(-1))
        self.btn_next.on_clicked(lambda evt: self.step(1))
        self.btn_play.on_clicked(lambda evt: self.toggle_play())
        self.btn_save.on_clicked(lambda evt: self._on_save_())
        self.slider.on_changed(self._on_slider)
        self.tb_new_points.on_submit(self._on_new_points_submit) 
        self.tb_ref_height.on_submit(self._on_height_submit) 


    def _draw_frame(self, i):
        frame_data = self.frames[i]
        frame_number = frame_data.get("frame_index", i)
        
        target_x, target_y, target_z = None, None, None 
        
        for p_id in range(self.max_people):
            x_raw, y_raw, z_raw = get_person_keypoints(frame_data, p_id)
            scat = self.all_scats[p_id]
            person_lines = self.all_lines[p_id]

            if x_raw is not None:
                x, y, z = np.array(x_raw), np.array(y_raw), np.array(z_raw) 
                
                if p_id == TARGET_PERSON_IDX:
                    target_x, target_y, target_z = x, y, z 
                
                x_sc, y_sc, z_sc = x * self.scale_factor, y * self.scale_factor, z * self.scale_factor
                valid_mask = np.isfinite(x_sc) & np.isfinite(y_sc) & np.isfinite(z_sc)
                x_valid, y_valid, z_valid = x_sc[valid_mask], y_sc[valid_mask], z_sc[valid_mask]
                
                scat._offsets3d = (x_valid, y_valid, z_valid)
                n = len(x_sc)
                for ln, a, b in person_lines:
                    if (a < n and b < n and np.isfinite(x_sc[a]) and np.isfinite(x_sc[b])):
                        ln.set_data_3d([x_sc[a], x_sc[b]], [y_sc[a], y_sc[b]], [z_sc[a], z_sc[b]])
                    else:
                        ln.set_data_3d([], [], [])
            else:
                scat._offsets3d = ([], [], [])
                for ln, a, b in person_lines:
                    ln.set_data_3d([], [], [])
        
        a, b = self.custom_line_points
        if target_x is not None and a < len(target_x) and b < len(target_x):
            if (np.isfinite(target_x[a]) and np.isfinite(target_x[b])):
                self.custom_line.set_data_3d(
                    [target_x[a] * self.scale_factor, target_x[b] * self.scale_factor], 
                    [target_y[a] * self.scale_factor, target_y[b] * self.scale_factor], 
                    [target_z[a] * self.scale_factor, target_z[b] * self.scale_factor]
                )
                kp_distance = distTwoPoints(target_x[a], target_y[a], target_z[a], 
                                            target_x[b], target_y[b], target_z[b])
                real_life_distance = kp_distance * self.sf_vertical
                mid_x = (target_x[a] + target_x[b]) / 2 * self.scale_factor
                mid_y = (target_y[a] + target_y[b]) / 2 * self.scale_factor
                mid_z = (target_z[a] + target_z[b]) / 2 * self.scale_factor
                self.custom_line_label.set_position((mid_x, mid_y, mid_z))
                self.custom_line_label.set_text(f"{real_life_distance:.2f} in.")
            else:
                self.custom_line.set_data_3d([], [], [])
                self.custom_line_label.set_text("")
        else:
            self.custom_line.set_data_3d([], [], [])
            self.custom_line_label.set_text("")

        self.ax.set_title(f"3D Multi-Pose — Frame {frame_number}")
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------------
    # 2.3. EVENT HANDLERS
    # ---------------------------------------------------------------------
    
    def _on_slider(self, val):
        self.i = int(val)
        self._draw_frame(self.i)

    def _on_timer(self):
        if not self.playing:
            return
        self.i = (self.i + 1) % self.num_frames
        self.slider.set_val(self.i) 

    def _on_save_(self):
        json_output_path = 'limbLengths.json'
        frame_data = self.frames[self.i]
        frame_number = frame_data.get("frame_index", self.i)
        
        x_raw, y_raw, z_raw = get_person_keypoints(frame_data, TARGET_PERSON_IDX)
        if x_raw is None:
            print(f"[SAVE ERROR] Target person (ID {TARGET_PERSON_IDX}) not detected.")
            return

        x, y, z = np.array(x_raw), np.array(y_raw), np.array(z_raw)
        n_kp = len(x)
        
        length_data = {
            "frame_index": frame_number,
            "target_person_id": TARGET_PERSON_IDX,
            "scaling_factor_in_per_meter": self.sf_vertical,
            "limb_lengths_inches": {}
        }

        for a_idx, b_idx, name in self.LIMB_SEGMENTS_TO_SAVE:
            if a_idx < n_kp and b_idx < n_kp:
                if (np.isfinite(x[a_idx]) and np.isfinite(x[b_idx])):
                    kp_dist = distTwoPoints(x[a_idx], y[a_idx], z[a_idx], 
                                            x[b_idx], y[b_idx], z[b_idx])
                    val = kp_dist * self.sf_vertical
                    length_data["limb_lengths_inches"][name] = round(val, 4)
                else:
                    length_data["limb_lengths_inches"][name] = "Keypoint_Missing"
            else:
                length_data["limb_lengths_inches"][name] = "Index_Out_of_Bounds"

        try:
            with open(json_output_path, 'w') as f:
                json.dump(length_data, f, indent=4)
            print(f"[SAVE SUCCESS] Saved to {json_output_path}")
        except Exception as e:
            print(f"[SAVE ERROR] {e}")

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
        self.slider.set_val(self.i)
    
    def _on_new_points_submit(self, text):
        try:
            a, b = map(int, text.split(","))
            max_idx = len(self.keypoint_names) - 1
            if a < 0 or b < 0 or a > max_idx or b > max_idx:
                 raise ValueError(f"Indices must be between 0 and {max_idx}.")
            self.custom_line_points = (a, b) 
            self._draw_frame(self.i) 
        except ValueError as e:
            print(f"[ERROR] {e}")
            start_kp, end_kp = self.custom_line_points
            self.tb_new_points.set_val(f"{start_kp},{end_kp}")
    
    def _on_height_submit(self, text):
        try:
            ref_inches = float(text)
            if ref_inches <= 0: raise ValueError("Must be positive.")
            frame_data = self.frames[self.i]
            x_raw, y_raw, z_raw = get_person_keypoints(frame_data, TARGET_PERSON_IDX)
            if x_raw is None: raise ValueError("Target not detected.")

            y = np.array(y_raw)
            kp_vertical = np.abs(y[REF_HEAD_IDX] - y[REF_FOOT_IDX])
            if kp_vertical == 0: raise ValueError("Zero vertical distance.")

            self.ref_height_inches = ref_inches
            self.sf_vertical = ref_inches / kp_vertical
            print(f"[SCALING] SF updated to {self.sf_vertical:.4f}")
            self._draw_frame(self.i)
        except ValueError as e:
            self.tb_ref_height.set_val(f"{self.ref_height_inches:.1f}")
            print(f"[ERROR] {e}")

    def run(self):
        self._draw_frame(self.i)
        plt.show()

if __name__ == '__main__':
    JSON_FILE_PATH = '/Users/franciscojimenez/Desktop/RecordingsFrom8to10/nybbc/3d_pose_data.json'
    player = MultiPose3DPlayer(json_path=JSON_FILE_PATH, fps=15)
    player.run()