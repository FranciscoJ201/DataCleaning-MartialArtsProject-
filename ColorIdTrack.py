import cv2
import numpy as np
import tkinter as tk
from tkinter import colorchooser
import colorsys

def start_tracking(hsv_target):
    """The main OpenCV tracking loop."""
    cap = cv2.VideoCapture(0)
    
    # Define color range (tolerance)
    # OpenCV H: 0-179, S: 0-255, V: 0-255
    #lower value is stricter
    tolerance = 14
    lower_bound = np.array([max(0, hsv_target[0] - tolerance), 70, 70])
    upper_bound = np.array([min(179, hsv_target[0] + tolerance), 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processing for tracking
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        # Clean up noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Draw the dot and crosshair
                    cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
                    cv2.putText(frame, "Tracking", (cX + 10, cY - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the two separate windows
        cv2.imshow("Live Tracking Feed", frame)
        cv2.imshow("Mask (What the Computer Sees)", mask)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

class ColorPickerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Selection Menu")
        self.root.geometry("300x250")
        self.selected_hsv = None

        tk.Label(root, text="Step 1: Choose Color", font=("Arial", 12)).pack(pady=10)
        
        self.color_btn = tk.Button(root, text="Open Color Wheel", command=self.pick_color)
        self.color_btn.pack(pady=5)

        self.preview = tk.Label(root, text="No color picked", width=20, height=2, bg="grey")
        self.preview.pack(pady=10)

        self.ready_btn = tk.Button(root, text="READY", state="disabled", 
                                   command=self.finish_gui, bg="green", fg="white")
        self.ready_btn.pack(pady=10)

    def pick_color(self):
        color = colorchooser.askcolor()
        if color[1]:
            self.preview.config(bg=color[1], text=color[1])
            # Convert RGB (0-255) to HSV for OpenCV
            r, g, b = [x/255.0 for x in color[0]]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            self.selected_hsv = (int(h*179), int(s*255), int(v*255))
            self.ready_btn.config(state="normal")

    def finish_gui(self):
        # Close the GUI and let the script move to the next step
        self.root.destroy()

if __name__ == "__main__":
    # Part 1: Run the GUI
    main_root = tk.Tk()
    app = ColorPickerGUI(main_root)
    main_root.mainloop()

    # Part 2: If a color was picked, run the OpenCV process
    if app.selected_hsv:
        print(f"Starting tracker with HSV: {app.selected_hsv}")
        start_tracking(app.selected_hsv)