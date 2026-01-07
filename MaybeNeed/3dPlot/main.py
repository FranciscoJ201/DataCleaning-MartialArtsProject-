from threeDimPlot import Pose3DPlayer
from LengthPrediction import calc_limb_lengths
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
from util import SMPL24_EDGES,lim, distTwoPoints,SMPL_BODY_PARTS
import numpy as np


def main():
    """
    Main function to handle file selection and user input for the 3D Pose Player.
    """
    # =========================================================================
    # 1. UI SETUP AND USER INPUT COLLECTION
    # =========================================================================
    
    # --- UI Initialization ---
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # --- Input: JSON File Path ---
    json_path = filedialog.askopenfilename(
        title="Select your 3D keypoint JSON file",
        filetypes=[("JSON files", "*.json")]
    )
    if not json_path:
        print("No file selected. Exiting.")
        return
    print(f"Selected JSON file: {json_path}")
    
    # --- Input: Target Person ID ---
    target_idx = simpledialog.askinteger(
        "Input",
        "Enter the target person's ID (e.g., 0, 1, 2...):",
        parent=root,
        minvalue=0
    )
    if target_idx is None:
        print("No target index provided. Exiting.")
        return
    print(f"Selected target ID: {target_idx}")

    # --- Input: Keypoints to Track (A, B) ---
    print(SMPL_BODY_PARTS)
    stringAB = simpledialog.askstring("Input","Enter the specific points you want to track (ex: 10,3 )",parent=root)
    a,b = stringAB.split(",")
    print(f"Selected points: {a,b} ")
    
    # --- Input: Fighter Height for Scaling ---
    fighter_height = simpledialog.askinteger("Input","Input an integer in INCHES (ex: 5 ft -> 60 inches)",parent=root,minvalue=0)
    print(f"Selected Height: {fighter_height}")


    # =========================================================================
    # 2. KEYPOINT DATA LOADING AND INITIAL CALCULATIONS
    # =========================================================================
    
    limb_lengths, coords = calc_limb_lengths(json_path, target_idx=target_idx)
    
    SF_vertical = 0.0 # Initialize Scaling Factor

    # --- Vertical Scaling Factor (SF_vertical) Calculation ---
    if coords is not None and limb_lengths is not None:
        # Unpack the Y-coordinates (vertical axis in this coordinate system)
        x, y, z = coords
        
        # Keypoint Indices for Head and Feet (based on SMPL-24)
        HEAD_IDX = 15
        LFOOT_IDX = 10
        RFOOT_IDX = 11
    
        # Check if necessary keypoints exist
        if all(idx < len(y) for idx in [HEAD_IDX, LFOOT_IDX, RFOOT_IDX]):
            # 1. Get the Y-coordinate for the head
            Y_Head = y[HEAD_IDX]
            # 2. Get the lowest Y-coordinate (the 'floor') from the two feet
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

    # --- Debugging Function: Real-World Length Calculation ---
    def REALWORLD():
        """
        Calculates and prints the real-world distance between the selected
        keypoints (a, b) using the calculated vertical scaling factor.
        (Intended for debugging/verification of SF_vertical).
        """
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


    # =========================================================================
    # 3. INITIALIZE AND RUN 3D VIEWER
    # =========================================================================

    # --- Player Initialization ---
    viewer = Pose3DPlayer(
        json_path=json_path,
        target_idx=target_idx,
        edges=SMPL24_EDGES,
        fps=30,
        fixed_limits=lim,
        auto_scale_margin=1.3,
        point_size=40,
        sf_vertical = SF_vertical # Pass the calculated scaling factor
    )


    # --- Setup and Launch ---
    viewer.connect_points(int(a),int(b)) # Connect the user-specified points for display/tracking
    REALWORLD()                           # Run the debug calculation
    viewer.run()                          # Start the Matplotlib 3D viewer


if __name__ == "__main__":
    main()