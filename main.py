from threeDimPlot import Pose3DPlayer
from LengthPrediction import calc_limb_lengths
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
from util import SMPL24_EDGES,lim, distTwoPoints
import numpy as np
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
    
    #4. Prompt for fighter height to determine the real world length from keypoint to keypoint
    fighter_height = simpledialog.askinteger("Input","Input an integer in INCHES (ex: 5 ft -> 60 inches)",parent=root,minvalue=0)
    print(f"Selected Height: {fighter_height}")
    
    
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
    limb_lengths, coords = calc_limb_lengths(json_path, target_idx=target_idx)












    #Real world length calculation:
    SF_vertical = 0.0


    #--------VERTICAL SCALING FACTOR-----------------
    if coords is not None and limb_lengths is not None:
        # Unpack the Z-coordinates (vertical axis)
        x, y, z = coords
        
        # Keypoint Indices for Head and Feet
        HEAD_IDX = 15
        LFOOT_IDX = 10
        RFOOT_IDX = 11
    
        # Check if necessary keypoints exist
        if all(idx < len(z) for idx in [HEAD_IDX, LFOOT_IDX, RFOOT_IDX]):
            # 1. Get the Z-coordinate for the head
            Y_Head = y[HEAD_IDX]
            # 2. Get the lowest Z-coordinate (the 'floor') from the two feet
            Y_min_Feet = np.min([y[LFOOT_IDX], y[RFOOT_IDX]])
            # 3. Calculate the Vertical Keypoint Height (Height_KP)
            Height_KP = np.abs( Y_Head - Y_min_Feet)
            # 4. Calculate the Scaling Factor (Inches per KP Unit)
            if Height_KP > 0:
                SF_vertical = fighter_height / Height_KP
                print(f"Calculated Scaling Factor VERTICAL (Inches/KP Unit): {SF_vertical:.4f}")
                print(f"\nCalculated Keypoint Vertical Height (Height_KP): {Height_KP:.2f}")
            else:
                print("Error: Calculated Height_KP is zero or negative. Cannot determine Scaling Factor VERTICAL.")
        else:
            print("Error: Head or Foot keypoints are missing in the data.")
    else:
        print("Could not retrieve keypoint coordinates for scaling factor calculation.")

    #--------REAL WORLD LENGTH CALCULATION-------------
    def REALWORLD():
        if coords is not None and SF_vertical > 0:
            x,y,z = coords
            idx_a = int(a)
            idx_b = int(b)

            try:
                kp_distance = distTwoPoints(x[idx_a], y[idx_a], z[idx_a],
                    x[idx_b], y[idx_b], z[idx_b]
                )
                real_life_distance = kp_distance * SF_vertical
                print("-" * 50)
                print(f"Keypoint Distance ({a}-{b}) in KP Units: {kp_distance:.2f}")
                print(f"Scaling Factor (Inches/KP Unit): {SF_vertical:.4f}")
                print(f"*** Real-Life Distance ({a}-{b}): {real_life_distance:.2f} inches ***")
                print("-" * 50)
            except IndexError:
                print("Error one or both keypoint indexs out of bounds")
        else:
            print("Mising coords or sf vert = 0 ")







    # right_leg_length = limb_lengths.get((16,18),0) + limb_lengths.get((18,20),0) + limb_lengths.get((20,22),0)
    # print(f"\nExample: Total estimated right leg length: {right_leg_length:.2f}")
    viewer.connect_points(int(a),int(b))
    REALWORLD()
    viewer.run()

if __name__ == "__main__":
    main()
    