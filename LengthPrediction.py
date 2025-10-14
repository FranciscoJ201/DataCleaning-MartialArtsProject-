from threeDimPersonPlot import Pose3DPlayer,SMPL24_EDGES,lim

#Key Aspects of the Script:
#1.) Length/Distance tracking between different body parts. 
#2.) 3d View Just like in the 3D single person plot script. 
#3.) Weight Tracking later...

#Requirements:
#1.) Reference length (height of one of the fighters preferably)
#2.) Keypoint JSON input 
JSON_PATH = '/Users/franciscojimenez/Desktop/3d.json'

def distTwoPoints(x1,x2,y1,y2,z1,z2):

viewer = Pose3DPlayer(
        json_path=JSON_PATH,
        target_idx=1,        # or an integer track id, e.g., 0 or 1
        edges=SMPL24_EDGES,
        fps=30,
        fixed_limits= lim,     
        auto_scale_margin=1.3,  # enlarge the autoscaled cube a bit
        point_size=40
    )
viewer.run()

