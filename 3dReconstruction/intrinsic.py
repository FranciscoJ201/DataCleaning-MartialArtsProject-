import cv2
import numpy as np
import glob
import json

def intrinsic_calb(path,outname):
    # --- Configuration ---
    # Number of inner corners on the checkerboard (e.g., a 7x10 square board has 6x9 inner corners)
    CHECKERBOARD = (6, 9)
    # Physical size of one square on the checkerboard (in mm, cm, or any consistent unit)
    SQUARE_SIZE_MM = 25.0  # <--- SET THIS TO THE REAL SIZE OF YOUR SQUARES

    # Define the termination criteria for the corner refinement (sub-pixel accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vectors to store 3D world points and 2D image points
    objpoints = []  # 3D points in real world space (constant for all images)
    imgpoints = []  # 2D points in image plane (pixel coordinates)

    # 3D points (real world coordinates)
    # Creates an array of shape (N, 3), where N is the number of corners (6*9=54)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    # Sets the X and Y coordinates based on the grid structure, scaled by the square size
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

    # --- Image Processing Loop ---
    # Searches for all JPG files in the current directory
    images = glob.glob(f'{outname}/*.jpg')

    # Variable to hold the shape of the grayscale image for the calibration function
    image_size = None 
    step = 0
    for filename in images:
        step += 1
        image = cv2.imread(filename)
        if image is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue
            
        # Convert color to grayscale for faster processing
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Store the image size once, which is needed for calibration
        if image_size is None:
            image_size = grayColor.shape[::-1] # Stores (width, height)
        
        # Find the corners of the chessboard
        ret, corners = cv2.findChessboardCorners(
            grayColor, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # If the desired number of corners is detected
        if ret:
            # Append the constant 3D world points list
            objpoints.append(objp)

            # Refine pixel coordinates to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners for verification
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
            
            print(f"Found corners in {filename} Frame:{step}")
            
        else:
            print(f"Did not find corners in {filename} Frame:{step}")


        cv2.imshow('img', image)
        cv2.waitKey(500) # Display for 500ms

    cv2.destroyAllWindows()

    # --- Calibration ---
    if not objpoints:
        print("Error: No images with detected checkerboards were found. Cannot calibrate.")
    else:
        # Perform the calibration
        # grayColor.shape[::-1] is correct for passing (width, height)
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None
        )

        # Calculate the Reprojection Error
        # This is a measure of the goodness of the calibration
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_reprojection_error = total_error / len(objpoints)

        # --- Output ---
    output_file = f"{outname}.json"
    calibration_data = {
        "mean_reprojection_error": float(mean_reprojection_error),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        # Optionally, you can include the image size, checkerboard size, and square size for reference
        "image_size": image_size, 
        "checkerboard_size": CHECKERBOARD,
        "square_size_mm": SQUARE_SIZE_MM,
        
        # NOTE: r_vecs and t_vecs are often large and may not be needed in the final JSON. 
        # If you need them, uncomment the following lines.
        "rotation_vectors": [r.tolist() for r in r_vecs],
        "translation_vectors": [t.tolist() for t in t_vecs]
    }
    with open(output_file, 'w') as f:
        # Use json.dump to save the entire dictionary
        json.dump(calibration_data, f, indent=4) 

    print(f"\nCalibration data successfully saved to {output_file}")


    print("\n--- Calibration Results ---")
    print(f"Mean Reprojection Error: {mean_reprojection_error:.4f} pixels (Lower is better)")
        
    print("\nCamera Matrix (K):")
    print(matrix)

    print("\nDistortion Coefficients (k1, k2, p1, p2, k3, ...):")
    print(distortion)

    # R and T vectors are lists of vectors, one for each successfully processed image.
    print(f"\nFound {len(r_vecs)} sets of Rotation and Translation Vectors.")