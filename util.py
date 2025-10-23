import numpy as np

def distTwoPoints(x1,y1,z1,x2,y2,z2):
    """
    This function is for calculating the euclidean distance between two 3D keypoints
    from an alphapose (or related software) json output
    """
    return np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 )
# ----------------------------
# Skeleton edges (COCO-17 default; swap to SMPL24 if you prefer)
# ----------------------------
COCO17_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,7),(7,9),
    (6,8),(8,10),(5,11),(6,12),
    (11,13),(13,15),(12,14),(14,16)
]
SMPL24_EDGES = [
    (0,1),(1,4),(4,7),(7,10),
    (0,2),(2,5),(5,8),(8,11),
    (0,3),(3,6),(6,9),(9,12),(12,15),
    (12,13),(13,16),(16,18),(18,20),(20,22),
    (12,14),(14,17),(17,19),(19,21),(21,23)
]

#CONSTS --------- 
lim = (-1.5,1.5)
#repaired on desktop^
# lim = None
#----------------
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
