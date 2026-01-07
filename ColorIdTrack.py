import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser
import colorsys

def start_tracking(hsv_target, sensitivity):
    cap = cv2.VideoCapture(0)
    
    # 1. Initialize the Background Subtractor (MOG2)
    
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

    lower_bound = np.array([max(0, hsv_target[0] - sensitivity), 50, 50])
    upper_bound = np.array([min(179, hsv_target[0] + sensitivity), 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 2. Apply Background Subtraction to find motion
        fg_mask = back_sub.apply(frame)
        
        # 3. Apply Color Filtering
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        # 4. COMBINE THEM: Pixel must be moving AND the right color
        # This removes stationary objects (chairs, walls) of the same color
        combined_mask = cv2.bitwise_and(color_mask, fg_mask)
        
        # 5. Clean up noise (Morphological opening)
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # 6. Tracking logic on the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 1000: 
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Moving Target", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Tracking (Motion + Color)", frame)
        cv2.imshow("Motion Mask", fg_mask)
        cv2.imshow("Combined Mask", combined_mask)
        
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


class ColorPickerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Settings")
        self.selected_hsv = None
        self.color_btn = tk.Button(root, text="1. Pick Color", command=self.pick_color)
        self.color_btn.pack(pady=10)
        tk.Label(root, text="2. Adjust Sensitivity (Threshold)").pack()
        self.sens_slider = tk.Scale(root, from_=5, to=50, orient="horizontal")
        self.sens_slider.set(20)
        self.sens_slider.pack(pady=10)
        self.ready_btn = tk.Button(root, text="READY", state="disabled", command=self.finish_gui)
        self.ready_btn.pack(pady=10)

    def pick_color(self):
        color = colorchooser.askcolor()
        if color[1]:
            r, g, b = [x/255.0 for x in color[0]]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            self.selected_hsv = (int(h*179), int(s*255), int(v*255))
            self.ready_btn.config(state="normal")

    def finish_gui(self):
        self.sensitivity = self.sens_slider.get()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColorPickerGUI(root)
    root.mainloop()

    if app.selected_hsv:
        start_tracking(app.selected_hsv, app.sensitivity)