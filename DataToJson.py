import numpy as np
import json
from util import SMPL_BODY_PARTS


data = [('0.jpg',np.float64(43.128048653258325)),('0.jpg',np.float64(41.128048653258325))]
fakeKPS= (10,3)
def createSegmentJson(data, custompoints):
    a, b = custompoints
    output_file = 'output_data.json'
    
    # Initialize a list to hold all the individual data dictionaries
    all_measurements = []
    key_name = f"{SMPL_BODY_PARTS[a]} - {SMPL_BODY_PARTS[b]}"
    for frame, np_float_value in data:
        inch_amount = float(np_float_value)
        output_data = {
            "frame": frame,
            key_name: inch_amount
        }
        all_measurements.append(output_data)
    try:
        with open(output_file, 'w') as f:
            # json.dump writes the entire list as a single JSON array
            json.dump(all_measurements, f, indent=4)
            # A final newline is optional, but keeps file formats clean
            f.write("\n")
            
        print(f"All {len(all_measurements)} data entries successfully written to {output_file}")
    except Exception as e:
        print(f"An error occurred while writing the JSON file: {e}")

createSegmentJson(data, fakeKPS)