import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame dimensions
    height, width, _ = frame.shape
    center_frame_x = width // 2
    center_frame_y = height // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Neon green range (adjustable)
    lower_neon_green = np.array([35, 80, 80])
    upper_neon_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_neon_green, upper_neon_green)

    # Clean mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        min_area = 200   # more forgiving
        
        if cv2.contourArea(largest_contour) > min_area:
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)

               

                print(f"Object Center: {center} | Offset: ({cX-center_frame_x}, {cY-center_frame_y})")

   
    cv2.imshow("Tracking Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
