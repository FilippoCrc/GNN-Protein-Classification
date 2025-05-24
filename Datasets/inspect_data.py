import json
import gzip
import os

def print_graph_info(graph_data, graph_index_in_file):
    """Helper function to print details of a single graph object."""
    print(f"\n--- Graph #{graph_index_in_file} (Sample from file) ---")
    if not isinstance(graph_data, dict):
        print(f"  Expected a dictionary for graph data, but got: {type(graph_data)}")
        if len(str(graph_data)) < 200: # Print if small
            print(f"  Content preview: {graph_data}")
        return

    print(f"  Keys present: {list(graph_data.keys())}")
    expected_keys = ['num_nodes', 'edge_list', 'node_feat', 'label'] # Common keys

    for key in expected_keys:
        if key in graph_data:
            value = graph_data[key]
            print(f"  Key '{key}':")
            print(f"    Type: {type(value)}")
            if isinstance(value, list):
                print(f"    Length: {len(value)}")
                if value: # If list is not empty
                    print(f"    First element type: {type(value[0])}")
                    if isinstance(value[0], list) and value[0]: # For lists of lists like edge_list
                            print(f"      First sub-element type: {type(value[0][0])}")
                    print(f"    Sample value (first 3 elements or less): {value[:3]}")
                else:
                    print(f"    Value: [] (empty list)")
            elif isinstance(value, (int, float, str, bool)):
                print(f"    Value: {value}")
            else:
                print(f"    Sample value (stringified, first 50 chars): {str(value)[:50]}")
        # For 'test' files, 'label' might be missing, which is fine.
        elif key == 'label':
            print(f"  Key '{key}': Not found (expected for train, optional for test).")
        # For other expected keys, if missing, it's a potential issue.
        # else:
        #     print(f"  Key '{key}': Not found in this graph object.")
    
    # List any other keys that are not in our 'expected_keys' list
    other_keys = [k for k in graph_data.keys() if k not in expected_keys]
    if other_keys:
        print(f"  Other keys found: {other_keys}")
        for key in other_keys:
            value = graph_data[key]
            # print(f"    Key '{key}': Type: {type(value)}, Sample: {str(value)[:50]}")


def fast_inspect_first_n_json_objects(filepath, num_objects_to_inspect=3):
    """
    Reads a .json.gz file containing a list of JSON objects, and prints details
    for the first 'num_objects_to_inspect' objects without loading the entire file.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    print(f"\n--- Fast Inspecting First {num_objects_to_inspect} Graph Objects from: {filepath} ---")
    objects_inspected_count = 0

    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            # 1. Find the start of the list '['
            char = ''
            while True:
                char = f.read(1)
                if not char:
                    print("Error: Reached end of file before finding list start '['.")
                    return
                if char == '[':
                    break
                if not char.isspace(): # If non-whitespace before '['
                    print(f"Warning: Unexpected characters before list start '[': '{char}'")


            # 2. Read and parse objects one by one
            current_object_buffer = ""
            brace_level = 0
            in_string = False # To correctly handle braces inside JSON strings
            string_delimiter = ''

            while objects_inspected_count < num_objects_to_inspect:
                char = f.read(1)
                if not char:
                    if current_object_buffer.strip():
                        print("Warning: Reached end of file while an object buffer was active.")
                    break # End of file

                current_object_buffer += char

                # Handle entering/exiting strings to ignore braces within them
                if char in ['"', "'"] and (len(current_object_buffer) == 1 or current_object_buffer[-2] != '\\'): # Start or end of string
                    if not in_string:
                        in_string = True
                        string_delimiter = char
                    elif in_string and char == string_delimiter:
                        is_escaped = False
                        # Check for escaped quote: count preceding backslashes
                        bs_count = 0
                        idx = len(current_object_buffer) - 2
                        while idx >= 0 and current_object_buffer[idx] == '\\':
                            bs_count += 1
                            idx -= 1
                        if bs_count % 2 == 0: # Not escaped, so it's a real end of string
                            in_string = False
                            string_delimiter = ''
                
                if not in_string:
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                        if brace_level == 0 and current_object_buffer.strip().startswith('{'):
                            # We likely have a complete object
                            try:
                                graph_data = json.loads(current_object_buffer)
                                objects_inspected_count += 1
                                print_graph_info(graph_data, objects_inspected_count)
                                current_object_buffer = "" # Reset for the next object
                                
                                # Skip trailing comma or whitespace before next object or end of list
                                while True:
                                    peek_char = f.read(1)
                                    if not peek_char: # EOF
                                        char = '' # Signal EOF to outer loop
                                        break
                                    if peek_char == ']': # End of list
                                        char = peek_char # Signal end to outer loop
                                        break
                                    if peek_char == '{': # Start of next object
                                        current_object_buffer = '{' # Prime buffer for next
                                        brace_level = 1 # As we've consumed the opening brace
                                        break
                                    if not peek_char.isspace() and peek_char != ',':
                                        print(f"Warning: Unexpected char '{peek_char}' after an object.")
                                        # Could try to prepend it to buffer if it's start of an object
                                        break 
                                
                            except json.JSONDecodeError as e:
                                print(f"  Error decoding JSON object: {e}")
                                print(f"  Problematic buffer (first 200 chars): {current_object_buffer[:200]}...")
                                # Attempt to recover: reset buffer and hope for the best
                                current_object_buffer = ""
                                brace_level = 0 # Should be 0 if error was after '}'
                            
                            if objects_inspected_count >= num_objects_to_inspect:
                                break
                
                if char == ']' and brace_level == 0 and not in_string: # End of the main list
                    if current_object_buffer.strip() not in ["]", ""]:
                        print(f"Warning: Content before final ']' not fully parsed: {current_object_buffer.strip()[:100]}")
                    break # Stop processing

            if objects_inspected_count == 0:
                print("No graph objects were successfully parsed from the beginning of the file.")
            elif objects_inspected_count < num_objects_to_inspect:
                print(f"Found and inspected only {objects_inspected_count} graph object(s).")

    except FileNotFoundError: # Already handled at the top
        pass
    except Exception as e:
        print(f"An unexpected error occurred while processing '{filepath}': {e}")

if __name__ == "__main__":
    DATASET_BASE_PATH = 'datasets'  # Your folder containing A, B, C, D
    SUBFOLDERS = ['A', 'B', 'C', 'D']
    # SUBFOLDERS = ['A'] # For quick testing one subfolder

    NUM_GRAPHS_TO_INSPECT_PER_FILE = 3 # As requested

    for subfolder_name in SUBFOLDERS:
        print(f"\n=========================================")
        print(f"===== Processing Subfolder: {subfolder_name} =====")
        print(f"=========================================")
        subfolder_path = os.path.join(DATASET_BASE_PATH, subfolder_name)

        train_file_path = os.path.join(subfolder_path, 'train.json.gz')
        test_file_path = os.path.join(subfolder_path, 'test.json.gz')

        if os.path.exists(train_file_path):
            fast_inspect_first_n_json_objects(train_file_path, num_objects_to_inspect=NUM_GRAPHS_TO_INSPECT_PER_FILE)
        else:
            print(f"\n--- File not found: {train_file_path} ---")

        if os.path.exists(test_file_path):
            fast_inspect_first_n_json_objects(test_file_path, num_objects_to_inspect=NUM_GRAPHS_TO_INSPECT_PER_FILE)
        else:
            print(f"\n--- File not found: {test_file_path} ---")

    print("\nFast partial inspection finished.")