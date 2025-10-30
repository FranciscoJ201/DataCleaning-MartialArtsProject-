import numpy as np
import json
from util import SMPL_BODY_PARTS
import os # Import the os module to check for file existence

data = [('0.jpg',np.float64(43.128048653258325)),('0.jpg',np.float64(41.128048653258325))]
fakeKPS= (10,3)

def createSegmentJson(data, custompoints,output_file):
    """
    Creates/Appends segment distance data to an existing JSON file.
    
    If the file exists, it reads the content, appends the new data, 
    and writes the updated list back. If the file does not exist, 
    it creates a new file.
    """
    a, b = custompoints
    
    
    # --- 1. Prepare new data entries ---
    new_measurements = []
    key_name = f"{SMPL_BODY_PARTS.get(a, f'Keypoint {a}')} - {SMPL_BODY_PARTS.get(b, f'Keypoint {b}')}"
    
    for frame, np_float_value in data:
        inch_amount = float(np_float_value)
        output_data = {
            "frame": frame,
            key_name: inch_amount
        }
        new_measurements.append(output_data)

    # --- 2. Load existing data or initialize an empty list ---
    all_measurements = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                # Load the existing list (or empty list if file is empty)
                file_content = f.read().strip()
                if file_content:
                    all_measurements = json.loads(file_content)
        except json.JSONDecodeError:
            print(f"Warning: Existing file {output_file} is corrupted or empty. Creating a new file.")
        except Exception as e:
            print(f"An error occurred while reading the JSON file: {e}. Starting fresh.")
    
    # --- 3. Append new data to the combined list ---
    all_measurements.extend(new_measurements)
    
    # --- 4. Write the entire updated list back to the file ---
    try:
        with open(output_file, 'w') as f:
            # json.dump writes the entire list as a single JSON array
            json.dump(all_measurements, f, indent=4)
            f.write("\n")
            
        print(f"Appended {len(new_measurements)} entries. Total data points now in {output_file}: {len(all_measurements)}")
    except Exception as e:
        print(f"A critical error occurred while writing the JSON file: {e}")

