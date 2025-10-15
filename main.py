from threeDimPlot import Pose3DPlayer
from LengthPrediction import calc_limb_lengths
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
from util import SMPL24_EDGES,lim
SMPL_BODY_PARTS = {
        0: "Pelvis",
        1: "Left Hip",
        2: "Right Hip",
        3: "Spine1",
        4: "Left Knee",
        5: "Right Knee",
        6: "Spine2",
        7: "Left Ankle",
        8: "Right Ankle",
        9: "Spine3",
        10: "Left Foot",
        11: "Right Foot",
        12: "Neck",
        13: "Left Collar",
        14: "Right Collar",
        15: "Head",
        16: "Left Shoulder",
        17: "Right Shoulder",
        18: "Left Elbow",
        19: "Right Elbow",
        20: "Left Wrist",
        21: "Right Wrist",
        22: "Left Hand",
        23: "Right Hand"
    }

def main():
    """
    Main function to handle file selection and user input for the 3D Pose Player.
    """
    # 1. File selection menu for JSON path
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Open file dialog to select the JSON file
    json_path = filedialog.askopenfilename(
        title="Select your 3D keypoint JSON file",
        filetypes=[("JSON files", "*.json")]
    )
    
    # Check if a file was selected
    if not json_path:
        print("No file selected. Exiting.")
        return
        
    print(f"Selected JSON file: {json_path}")
    
    # 2. Prompt box for target_idx
    target_idx = simpledialog.askinteger(
        "Input",
        "Enter the target person's ID (e.g., 0, 1, 2...):",
        parent=root,
        minvalue=0
    )
     # Check if a target index was provided
    if target_idx is None:
        print("No target index provided. Exiting.")
        return
    print(f"Selected target ID: {target_idx}")

    # 3. Prompt Box for target keypoints
    print(SMPL_BODY_PARTS)
    stringAB = simpledialog.askstring("Input","Enter the specific points you want to track (ex: 10,3 )",parent=root)
    a,b = stringAB.split(",")
    print(f"Selected points: {a,b} ")
    
    
    
    # Instantiate and run the 3D Pose Player with user inputs
    viewer = Pose3DPlayer(
        json_path=json_path,
        target_idx=target_idx,
        edges=SMPL24_EDGES,
        fps=30,
        fixed_limits=lim,
        auto_scale_margin=1.3,
        point_size=40
    )
    limb_lengths = calc_limb_lengths(json_path, target_idx=target_idx)

    right_leg_length = limb_lengths.get((16,18),0) + limb_lengths.get((18,20),0) + limb_lengths.get((20,22),0)
    print(f"\nExample: Total estimated right leg length: {right_leg_length:.2f}")
    
    viewer.connect_points(int(a),int(b))
    viewer.run()

if __name__ == "__main__":
    main()
    