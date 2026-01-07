from threeDimPlot import load_frames,select_person_entry,get_xyz_from_entry
import numpy as np
from util import distTwoPoints,SMPL24_EDGES
#Key Aspects of the Script:
#1.) Length/Distance tracking between different body parts. 
#2.) 3d View Just like in the 3D single person plot script. 
#3.) Weight Tracking later...

#Requirements:
#1.) Reference length (height of one of the fighters preferably)
#2.) Keypoint JSON input 
JSON_PATH = '/Users/franciscojimenez/Desktop/3d.json'


# MODIFIED: Now returns limb_lengths AND xyz coordinates (x, y, z)
def calc_limb_lengths(json_path, target_idx, edges = SMPL24_EDGES):
    """
    Calculates the lengths of limbs for a specific person (idx) in a JSON file
    and returns the coordinates of the first valid frame.
    
    Args:
        json_path (str): The path to the JSON file with keypoint data.
        target_idx (int): The track ID of the person to analyze.
        edges (list): A list of tuples defining the skeleton edges.
        
    Returns:
        tuple: (dict of limb lengths, tuple of keypoint coordinates (x, y, z))
               Returns (None, None) on failure.
    """
    #load the frames from the json file
    keys, frames = load_frames(json_path)

    #----ERROR HANDLING-------

    #get the first frame with VALID data
    first_frame_key = None
    x, y, z = None, None, None
    for key in keys:
        entries = frames[key]
        entry = select_person_entry(entries, target_idx)
        x,y,z = get_xyz_from_entry(entry)
        if x is not None:
            first_frame_key = key
            break

    #if no usable data return
    if first_frame_key is None:
        print("No frame with valid keypoint data for selected target")
        return None, None 
    
    # Get keypoint coords for the target in the first valid frame
    # (x, y, z are already set from the loop above)

    # Calculate and store distances for each edge
    lengths = {}
    for edge in edges:
        a,b = edge
        if a < len(x) and b < len(x):
            try:
                # Use the existing limb calculation
                distance = distTwoPoints(x[a],y[a],z[a],x[b],y[b],z[b])
                lengths[edge] = distance
                # NOTE: Commenting out the print for brevity in the final output
                # print(f"Distance between keypoints {a} and {b}: {distance:.2f}")
            except IndexError:
                print(f"Error Keypoints {a} or {b} are out of bounds for the skeleton.")
                
    return lengths, (x, y, z)