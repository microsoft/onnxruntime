import numpy as np
import argparse
import os
import glob

def convert_raw_to_npy(raw_file_path, output_npy_path, data_type_str, shape_str, byte_order=None):
    """
    Converts a raw tensor file to a .npy file.
    """
    try:
        try:
            if byte_order:
                dt = np.dtype(data_type_str)
                tensor_dtype = dt.newbyteorder(byte_order)
            else:
                tensor_dtype = np.dtype(data_type_str)
        except TypeError as e:
            print(f"Error: Invalid data type string '{data_type_str}'. See NumPy documentation for valid types.")
            print(f"Details: {e}")
            return False

        try:
            # Handle single dimension case (e.g., "512" or "512,")
            shape_parts = [s.strip() for s in shape_str.split(',') if s.strip()]
            if not shape_parts:
                raise ValueError("Shape string cannot be empty.")
            
            tensor_shape = tuple(map(int, shape_parts))
            if not tensor_shape or any(s <= 0 for s in tensor_shape):
                raise ValueError("Shape dimensions must be positive integers.")
        except ValueError as e:
            print(f"Error: Invalid shape string '{shape_str}'. Must be comma-separated positive integers (e.g., '100,50,3').")
            print(f"Details: {e}")
            return False

        if not os.path.exists(raw_file_path):
            print(f"Error: Input file '{raw_file_path}' not found.")
            return False

        try:
            with open(raw_file_path, 'rb') as f:
                raw_data = f.read()
        except IOError as e:
            print(f"Error: Cannot read file '{raw_file_path}': {e}")
            return False

        actual_size = len(raw_data)
        if actual_size == 0:
            print(f"Error: Input file '{raw_file_path}' is empty.")
            return False

        expected_elements = np.prod(tensor_shape, dtype=np.int64)
        expected_size = expected_elements * tensor_dtype.itemsize

        # Validate file size
        if actual_size < expected_size:
            print(f"Error: Input file '{raw_file_path}' (size: {actual_size} bytes) "
                  f"is too small for the specified shape {tensor_shape} and dtype {tensor_dtype} "
                  f"(requires {expected_size} bytes).")
            print("Cannot proceed with conversion. Please verify shape and dtype.")
            return False
        elif actual_size > expected_size:
            print(f"Warning: Input file '{raw_file_path}' (size: {actual_size} bytes) "
                  f"is larger than required for shape {tensor_shape} and dtype {tensor_dtype} "
                  f"(expected {expected_size} bytes).")
            print(f"Only the first {expected_size} bytes will be used.")
        
        # Convert raw data to numpy array
        try:
            numpy_array = np.frombuffer(raw_data, dtype=tensor_dtype, count=expected_elements)
            
            reshaped_array = numpy_array.reshape(tensor_shape)
        except ValueError as e:
            print(f"Error during array conversion: {e}")
            print("This might be due to incorrect shape, data type, or incompatible raw data format.")
            return False

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_npy_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"Error: Cannot create output directory '{output_dir}': {e}")
                return False

        try:
            np.save(output_npy_path, reshaped_array)
        except IOError as e:
            print(f"Error: Cannot write to output file '{output_npy_path}': {e}")
            return False

        print(f"Successfully converted '{raw_file_path}' to '{output_npy_path}'")
        print(f"  Data Type: {reshaped_array.dtype}")
        print(f"  Shape: {reshaped_array.shape}")
        print(f"  Size: {reshaped_array.nbytes} bytes")
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Details: dtype='{data_type_str}', shape='{shape_str}', byte_order='{byte_order}'")
        return False

def batch_convert_raw_to_npy(folder_path, data_type_str, shape_str, file_pattern="*", byte_order=None, delete_originals=True):
    """
    Converts all raw files in a folder to .npy format and optionally deletes originals.
    
    Args:
        folder_path: Path to the folder containing raw files
        data_type_str: NumPy data type string
        shape_str: Comma-separated tensor shape
        file_pattern: Pattern to match files (default: "*" for all files)
        byte_order: Byte order specification
        delete_originals: Whether to delete original files after successful conversion
    
    Returns:
        tuple: (successful_conversions, failed_conversions, deleted_files)
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return 0, 0, 0
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        return 0, 0, 0
    
    # Find all files matching the pattern
    search_pattern = os.path.join(folder_path, file_pattern)
    raw_files = glob.glob(search_pattern)
    
    # Filter out any .npy files to avoid processing them
    raw_files = [f for f in raw_files if not f.endswith('.npy')]
    
    if not raw_files:
        print(f"No raw files found in '{folder_path}' matching pattern '{file_pattern}'")
        return 0, 0, 0
    
    print(f"Found {len(raw_files)} files to convert in '{folder_path}'")
    print(f"Data type: {data_type_str}, Shape: {shape_str}")
    if byte_order:
        print(f"Byte order: {byte_order}")
    print("-" * 50)
    
    successful_conversions = 0
    failed_conversions = 0
    deleted_files = 0
    files_to_delete = []
    
    for raw_file in raw_files:
        base, ext = os.path.splitext(raw_file)
        output_file = base + ".npy"
        
        print(f"\nProcessing: {os.path.basename(raw_file)}")
        
        success = convert_raw_to_npy(raw_file, output_file, data_type_str, shape_str, byte_order)
        
        if success:
            successful_conversions += 1
            files_to_delete.append(raw_file)
        else:
            failed_conversions += 1
            print(f"Failed to convert '{raw_file}'")
    
    # Delete original files if requested and conversions were successful
    if delete_originals and files_to_delete:
        print(f"\n{'='*50}")
        print("Deleting original raw files...")
        
        for raw_file in files_to_delete:
            try:
                os.remove(raw_file)
                deleted_files += 1
                print(f"Deleted: {os.path.basename(raw_file)}")
            except OSError as e:
                print(f"Warning: Could not delete '{raw_file}': {e}")
    
    print(f"\n{'='*50}")
    print("BATCH CONVERSION SUMMARY:")
    print(f"  Total files processed: {len(raw_files)}")
    print(f"  Successful conversions: {successful_conversions}")
    print(f"  Failed conversions: {failed_conversions}")
    if delete_originals:
        print(f"  Files deleted: {deleted_files}")
    print(f"{'='*50}")
    
    return successful_conversions, failed_conversions, deleted_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw tensor files to .npy format. Can process single files or entire folders.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Make the first argument optional and add folder mode
    parser.add_argument(
        "input_path", 
        help="Path to input raw tensor file OR folder containing raw files."
    )
    parser.add_argument(
        "dtype", 
        help="NumPy data type string (e.g., 'float32', 'int16', 'uint8')."
    )
    parser.add_argument(
        "shape", 
        help="Comma-separated tensor shape (e.g., '100,100,3' or '512')."
    )
    
    # Options for single file mode
    parser.add_argument(
        "-o", "--output_file",
        help="Optional: Path to the output .npy file (single file mode only). \n"
             "If not provided, it defaults to the input filename with a .npy extension."
    )
    
    # Options for batch mode
    parser.add_argument(
        "-f", "--folder_mode",
        action="store_true",
        help="Enable folder mode to process all files in the specified folder."
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*",
        help="File pattern to match in folder mode (default: '*' for all files)."
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Don't delete original raw files after successful conversion (folder mode only)."
    )
    
    parser.add_argument(
        "-b", "--byte_order",
        choices=['<', '>'],
        help="Optional: Specify byte order. '<' for little-endian, '>' for big-endian. \n"
             "Defaults to system's native byte order if not specified."
    )

    args = parser.parse_args()

    if args.folder_mode or os.path.isdir(args.input_path):
        success_count, fail_count, delete_count = batch_convert_raw_to_npy(
            args.input_path, 
            args.dtype, 
            args.shape, 
            args.pattern,
            args.byte_order,
            not args.no_delete
        )
        
        exit(0 if fail_count == 0 else 1)
        
    else:
        output_file_path = args.output_file
        if not output_file_path:
            base, ext = os.path.splitext(args.input_path)
            output_file_path = base + ".npy"

        success = convert_raw_to_npy(args.input_path, output_file_path, args.dtype, args.shape, args.byte_order)
        
        exit(0 if success else 1)