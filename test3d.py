import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# The user-specified skeleton edges
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
]

def plot_3d_pose(triangulate_json_path, save_image_path="3d_pose_plot.png", edges=SKELETON_EDGES):
    """
    Creates a simple 3D scatter plot of the first reconstructed pose from the 
    triangulate.py output JSON file, with improved axis scaling.

    Args:
        triangulate_json_path (str): Path to the JSON file output by triangulate.py.
        save_image_path (str): Path to save the output plot image.
        edges (list): A list of (keypoint_index_A, keypoint_index_B) tuples
                      to draw lines between.
    """
    try:
        with open(triangulate_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    if not data or not data[0].get("keypoints_3d_mm"):
        print("Error: JSON data is empty or missing 'keypoints_3d_mm'.")
        return

    # 1. Get the 3D keypoints for the first detected pose
    keypoints_4d = np.array(data[0]["keypoints_3d_mm"], dtype=np.float64)
    x = keypoints_4d[:, 0]
    y = keypoints_4d[:, 1]
    z = keypoints_4d[:, 2]

    # Calculate initial values for the diagnostic check (before custom limit calculation)
    all_coords = np.concatenate([x, y, z])
    max_val = np.max(all_coords)
    min_val = np.min(all_coords)
    span_check = (max_val - min_val) * 1.1 
    center_check = (max_val + min_val) / 2.0
    lo_check = center_check - span_check / 2.0
    hi_check = center_check + span_check / 2.0

    # --- Diagnostic Check (Integrated as requested) ---
    print("\n--- Diagnostic Check ---")
    print(f"Total keypoints found: {len(x)}")
    print(f"X-range: Min={x.min():.2f}, Max={x.max():.2f}")
    print(f"Y-range: Min={y.min():.2f}, Max={y.max():.2f}")
    print(f"Z-range: Min={z.min():.2f}, Max={z.max():.2f}")
    print(f"Coordinate Max Value: {max_val:.2f}, Min Value: {min_val:.2f}")
    print(f"Calculated Plot Limits (Default): {lo_check:.2f} to {hi_check:.2f}")
    print("------------------------")
    
    # 2. Setup Figure and Axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3. Plot the Keypoints (Scatter Plot)
    ax.scatter(x, y, z, c='blue', marker='o', s=50)

    # 4. Plot the Connections (Edges)
    for a, b in edges:
        if a < len(x) and b < len(x):
            ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]], c='red', linewidth=2)

    # 5. Set Axes Limits with TIGHT, centered auto-scaling (Fixes the zoom issue)
    
    # Calculate the true range and center for each axis
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    max_range = max(x_range, y_range, z_range)
    
    center_x = (x.max() + x.min()) / 2.0
    center_y = (y.max() + y.min()) / 2.0
    center_z = (z.max() + z.min()) / 2.0

    # Calculate the half-limit (radius) including a small 5% margin
    margin = max_range * 0.05 
    limit = max_range / 2.0 + margin 

    # Apply the same limit to all three axes, centered on the data's mean position
    ax.set_xlim(center_x - limit, center_x + limit)
    ax.set_ylim(center_y - limit, center_y + limit)
    ax.set_zlim(center_z - limit, center_z + limit)
    
    print(f"--- Applied Tighter Limits: {center_x - limit:.2f} to {center_x + limit:.2f} (based on max range {max_range:.2f}) ---")
    
    # Labels
    ax.set_xlabel("X Coordinate (mm)")
    ax.set_ylabel("Y Coordinate (mm)")
    ax.set_zlabel("Z Coordinate (mm)")
    
    pose_index = data[0].get('pose_index', 0)
    ax.set_title(f"3D Pose Reconstruction (Pose Index: {pose_index})")
    
    # Set a standard viewing angle
    ax.view_init(elev=10, azim=-60)
    
    # 6. Save the plot
    plt.tight_layout()
    plt.savefig(save_image_path)
    plt.close(fig)
    
    print(f"\n3D plot successfully saved to {save_image_path}")
    
# Example Usage (replace 'path/to/triangulate_output.json' with your actual file)
if __name__ == '__main__':
    plot_3d_pose('/Users/franciscojimenez/Desktop/DataCleaning-MartialArtsProject-/3d_pose_reconstruction.json')