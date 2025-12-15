from ultralytics import YOLO
import os

# --- CONFIGURATION ---
MODEL_NAME = 'yolo11x-pose.pt' 
# MODEL_NAME = 'yolov11n-pose.pt' # Use this if you want to try v11
ENGINE_FILE = MODEL_NAME.replace('.pt', '.engine')

if os.path.exists(ENGINE_FILE):
    print(f"TensorRT engine '{ENGINE_FILE}' already exists. Skipping export.")
else:
    print(f"Loading PyTorch model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"Exporting model to TensorRT engine (FP16 precision) as {ENGINE_FILE}...")
    # 'half=True' uses FP16 (recommended for T4)
    # 'device=0' ensures the export process runs on the first GPU
    model.export(format='engine', half=True, device=0)
    
    if os.path.exists(ENGINE_FILE):
        print(f"\n✅ SUCCESS: TensorRT engine created at: {os.path.abspath(ENGINE_FILE)}")
        print("You can now run your main RealSense script.")
    else:
        print("\n❌ FAILED: The engine file was not created. Check your CUDA/TensorRT setup.")