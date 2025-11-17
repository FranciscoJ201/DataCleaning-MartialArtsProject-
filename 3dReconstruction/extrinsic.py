import cv2 
import numpy as np
import json
import glob

def extrinstic_calb(path1,path2,show):
    #have these autofill from json later
    
    INTRINSIC_FILE_1 = f'{path1}.json'
    INTRINSIC_FILE_2 = f'{path2}.json'
    with open(INTRINSIC_FILE_1, 'r') as f:
            data1 = json.load(f)
    with open(INTRINSIC_FILE_2, 'r') as f:
            data2 = json.load(f)

    if data1["checkerboard_size"] == data2["checkerboard_size"]:
        CHECKERBOARD = (data1['checkerboard_size'][0],data1['checkerboard_size'][1])
        SQUARE_SIZE_MM = data1['square_size_mm'] 
        print(f'YES KING: {CHECKERBOARD}')
    else:
        print('KABOOM')
        exit()
    
    LEFT_IMAGES_PATH = f'{path1}/main*.jpg'
    RIGHT_IMAGES_PATH = f'{path2}/helper*.jpg'


    def load_intrinsics(file_path):
        """Loads Camera Matrix (K) and Distortion Coeffs (D) from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        K = np.array(data["camera_matrix"], dtype=np.float64)
        D = np.array(data["distortion_coefficients"], dtype=np.float64)
        image_size = tuple(data["image_size"])
        return K, D, image_size
    
    try:
        K1,D1,image_size = load_intrinsics(INTRINSIC_FILE_1)
        K2,D2, _ = load_intrinsics(INTRINSIC_FILE_2)
    except FileNotFoundError as e:
        print(f"ERROR: Required intrinsic file not found. {e}")
        exit()
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    #3d points (real world coordinates)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    #Vectors for 3d world points and 2d image points storage
    objpoints = []
    imgpoints1 = []
    imgpoints2 = []

    #---------- Image Processing Loop ------------------

    left_images = sorted(glob.glob(LEFT_IMAGES_PATH))
    right_images = sorted(glob.glob(RIGHT_IMAGES_PATH))

    if len(left_images) != len(right_images):
        print('Error: mismatched number of images between right and left folders')
        exit()
    print(f"Found {len(left_images)} image pairs. Processing begins...")


    #iterate thru the image pairs
    for filename1, filename2 in zip(left_images, right_images):
        img1 = cv2.imread(filename1)
        img2 = cv2.imread(filename2)

        if img1 is None or img2 is None:
            print(f"Warning: Could not read image pair ({filename1}, {filename2}). Skipping.")
            continue
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)
        ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)

        #If corners are found in both images: 
        if ret1 and ret2: 
            #refine corners 
            corners1_refined= cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2_refined= cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            #store points
            objpoints.append(objp)
            imgpoints1.append(corners1_refined)
            imgpoints2.append(corners2_refined)

            print(f"Found corners in pair: {filename1} and {filename2}")

            if show == True:
                img1 = cv2.drawChessboardCorners(img1, CHECKERBOARD, corners1_refined, ret1)
                img2 = cv2.drawChessboardCorners(img2, CHECKERBOARD, corners2_refined, ret2)
                cv2.imshow('Left View', img1)
                cv2.imshow('Right View', img2)
                cv2.waitKey(100)
        else:
            print(f"Skipping pair: Corners not found in one or both images {filename1}, {filename2}")

    cv2.destroyAllWindows()


    #------------- Stereo Calibration -----------
    #finding r and t

    if not objpoints:
        print('Error: No valid image pairs with detected checkerboards found. Cannot calibrate')
        exit()


    print(f'Starting Stereo Calibration with {len(objpoints)} valid pairs')

    #Stereo calibration 
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints1, imgpoints2,
        K1, D1, K2, D2,
        image_size,
        criteria=criteria,
        # Crucial flag: We fix the intrinsics (K and D) to only solve for R and T
        flags=cv2.CALIB_FIX_INTRINSIC 
    )
    

    #--------- Output Results --------------

    stereo_error = ret
    output_file = 'stereo_calibration.json'
    stereo_data = {
        "stereo_reprojection_error": float(stereo_error),
        "rotation_matrix_R1_to_R2": R.tolist(),
        "translation_vector_T1_to_T2": T.tolist(),
        "image_size": image_size, 
        "checkerboard_size": CHECKERBOARD,
        "square_size_mm": SQUARE_SIZE_MM
    }

    with open(output_file, 'w') as f:
        json.dump(stereo_data, f, indent=4)

    print(f"\nStereo Calibration data successfully saved to {output_file}")
    print("\n--- Stereo Calibration Results ---")
    print(f"Reprojection Error: {stereo_error:.4f} pixels (Lower is better)")
    print("\nRotation Matrix (R) - Cam 1 to Cam 2:")
    print(R)
    print("\nTranslation Vector (T) - Cam 1 to Cam 2 (in mm):")
    print(T)      