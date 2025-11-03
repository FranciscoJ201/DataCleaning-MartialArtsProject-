import json

def analyze_hierarchy(data, indent=0, key_name="ROOT"):
    """
    Recursively traverses a JSON structure and prints its hierarchy.

    Args:
        data: The current JSON object, array, or value.
        indent: The current indentation level (depth in the structure).
        key_name: The key used to access the current data structure.
    """
    # Create the indentation string
    indent_str = "    " * indent
    
    # Check if the data is a JSON object (Python dictionary)
    if isinstance(data, dict):
        print(f"{indent_str}KEY: {key_name} (Object)")
        # Iterate through the keys and call the function recursively for each value
        for key, value in data.items():
            analyze_hierarchy(value, indent + 1, key)
            
    # Check if the data is a JSON array (Python list)
    elif isinstance(data, list):
        # Print the key/label for the list, followed by the array structure marker
        print(f"{indent_str}KEY: {key_name} (Array [{len(data)} items])")
        
        # We only look at the first item's structure to define the array's template
        if data:
            # Use an index placeholder '[...]' to show it's a repeatable element structure
            print(f"{indent_str}    [...]") 
            
            # Recursively analyze the first element to define the structure template
            # We use a placeholder key_name "Element" for the inner structure
            analyze_hierarchy(data[0], indent + 2, "Element")
        else:
            # Handle empty lists
            print(f"{indent_str}    (Empty List)")

    # The data is a primitive value (string, number, boolean, or null)
    else:
        # Determine the type of the primitive value
        value_type = type(data).__name__
        if data is None:
             value_type = "null"
             
        print(f"{indent_str}KEY: {key_name} ({value_type})")
        
# --- Main Execution Block ---

# Path to the JSON file you want to analyze
FILE_PATH = "/Users/franciscojimenez/Desktop/3d.json"

print("--- JSON HIERARCHY ANALYSIS ---")
print("Note: For arrays, only the structure of the first element is shown.")
print(f"Attempting to read file: {FILE_PATH}")
print("-------------------------------\n")

try:
    # To read from a file, you must use 'open()' to read the file content
    # and then use 'json.load()' (without the 's') to parse the stream.
    with open(FILE_PATH, 'r') as f:
        data_object = json.load(f)
        
    # Start the recursive analysis
    analyze_hierarchy(data_object)

except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}. Please check the file path.")
except json.JSONDecodeError as e:
    # This error occurs if the file content is not valid JSON
    print(f"Error decoding JSON in file '{FILE_PATH}': {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
