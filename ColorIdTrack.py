import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser
import colorsys

def start_tracking(hsv_target, sensitivity):
    cap = cv2.VideoCapture(0)
    
    # Apply the sensitivity from the slider to the Hue range
    lower_bound = np.array([max(0, hsv_target[0] - sensitivity), 50, 50])
    upper_bound = np.array([min(179, hsv_target[0] + sensitivity), 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret: break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        # Tracking logic
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                # --- CHANGED LOGIC START ---
                # Get the coordinates for the bounding box
                x, y, w, h = cv2.boundingRect(largest)
                
                # Draw the rectangle: (image, start_point, end_point, color, thickness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Optional: Add a label above the box
                cv2.putText(frame, "Largest Instance", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # --- CHANGED LOGIC END ---

        cv2.imshow("Tracking", frame)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# ... (The rest of your ColorPickerGUI class and __main__ remain exactly the same)
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