from threeDimPersonPlot import Pose3DPlayer,SMPL24_EDGES,lim,load_frames,select_person_entry,get_xyz_from_entry
import numpy as np
#Key Aspects of the Script:
#1.) Length/Distance tracking between different body parts. 
#2.) 3d View Just like in the 3D single person plot script. 
#3.) Weight Tracking later...

#Requirements:
#1.) Reference length (height of one of the fighters preferably)
#2.) Keypoint JSON input 
JSON_PATH = '/Users/franciscojimenez/Desktop/3d.json'

def distTwoPoints(x1,y1,z1,x2,y2,z2):
    """
    This function is for calculating the euclidean distance between two 3D keypoints
    from an alphapose (or related software) json output
    sqrt[(x2-x1)^2 + (y2-y1)^2 + ..... dimensions
    """
    return np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )

def calc_limb_lengths(json_path, target_idx, edges = SMPL24_EDGES):
    """
    Calculates the lengths of limbs for a specific person (idx) in a JSON file.
    
    Args:
        json_path (str): The path to the JSON file with keypoint data.
        target_idx (int): The track ID of the person to analyze.
        edges (list): A list of tuples defining the skeleton edges.
        
    Returns:
        dict: A dictionary mapping edge tuples to their calculated lengths.
    """
    #load the frames from the json file
    keys, frames = load_frames(json_path)

    #----ERROR HANDLING-------

    #get the first frame with VALID data
    first_frame_key = None
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
        return 
    
    #Get keypoint coords for the target in the first valid frame
    entry = select_person_entry(frames[first_frame_key], target_idx)
    x,y,z = get_xyz_from_entry(entry)

    #second check...
    if x is None:
        print(f"Cant get keypoints for target {target_idx}")
        return
    
    #calculate and store distances for each edge
    lengths = {}
    for edge in edges:
        a,b = edge
        if a < len(x) and b < len(x):
            try:
                distance = distTwoPoints(x[a],y[a],z[a],x[b],y[b],z[b])
                lengths[edge] = distance
                print(f"Distance between keypoints {a} and {b}: {distance:.2f}")
            except IndexError:
                print(f"Error Keypoints {a} or {b} are out of bounds for the skeleton.")
    return lengths


# if __name__ == '__main__':

#     trgt_idx = 1

#     limb_lengths = calc_limb_lengths(JSON_PATH, target_idx=trgt_idx)

#     right_leg_length = limb_lengths.get((16,18),0) + limb_lengths.get((18,20),0) + limb_lengths.get((20,22),0)
#     print(f"\nExample: Total estimated right leg length: {right_leg_length:.2f}")


#     viewer = Pose3DPlayer(
#             json_path=JSON_PATH,
#             target_idx=trgt_idx,        # or an integer track id, e.g., 0 or 1
#             edges=SMPL24_EDGES,
#             fps=30,
#             fixed_limits= lim,     
#             auto_scale_margin=1.3,  # enlarge the autoscaled cube a bit 1.3
#             point_size=40
#         )
#     viewer.connect_points(10,3)
#     viewer.run()
    
