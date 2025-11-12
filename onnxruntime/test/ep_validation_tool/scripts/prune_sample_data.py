
"""
Prune files that do not have common suffixes across input folders.

This script finds .npy files with matching numeric suffixes across multiple folders
and removes files that don't have suffixes common to all folders.

Usage Examples:
    # Basic usage with 2 folders:
    python prune_sample_data.py "A\input_1" "A\input_2"
    
    # Multiple folders:
    python prune_sample_data.py "A\input_1" "A\input_2" "A\input_3" "A\input_4"
    
    # Using absolute paths:
    python prune_sample_data.py "C:\path\to\data\input_1" "C:\path\to\data\input_2"
    
    # With different folder names:
    python prune_sample_data.py "data\train" "data\test" "data\validation"
    
    # Getting help:
    python prune_sample_data.py --help

Expected File Structure:
    Each folder should contain .npy files named like {folder_name}_{number}.npy
    Example:
        A/
        ├── input_1/
        │   ├── input_1_001.npy
        │   ├── input_1_002.npy
        │   └── input_1_005.npy
        └── input_2/
            ├── input_2_001.npy
            ├── input_2_003.npy
            └── input_2_005.npy
    
    Running the script would keep only files with suffixes 001 and 005 (common to both)
    and remove input_1_002.npy and input_2_003.npy.
"""

import os
import re
import argparse
import sys

# Regular expression to extract numeric suffix from filenames
# This pattern will match any input folder name followed by underscore and number
pattern = re.compile(r'^(.+)_(\d+)\.npy$')

# Function to extract numeric suffixes from filenames in a folder
def get_suffixes(folder, expected_prefix):
    suffixes = set()
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
        return suffixes
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            file_prefix = match.group(1)
            # Check if the file prefix matches the expected folder prefix
            if file_prefix == expected_prefix:
                suffixes.add(match.group(2))
    return suffixes

# Function to prune files not having common suffixes
def prune_folder(folder, expected_prefix, common_suffixes):
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
        return
        
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            file_prefix = match.group(1)
            if file_prefix == expected_prefix:
                suffix = match.group(2)
                if suffix not in common_suffixes:
                    file_path = os.path.join(folder, filename)
                    os.remove(file_path)
                    print(f"Removed: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Prune files that do not have common suffixes across input folders')
    parser.add_argument('folders', nargs='+', help='Input folder paths to process')
    
    args = parser.parse_args()
    input_folders = args.folders
    
    if len(input_folders) < 2:
        print("Error: At least 2 input folders are required")
        sys.exit(1)
    
    # Dictionary to store suffixes for each folder
    folder_suffixes = {}
    
    # Extract folder name from path to use as prefix
    for folder in input_folders:
        folder_name = os.path.basename(folder.rstrip(os.sep))
        suffixes = get_suffixes(folder, folder_name)
        folder_suffixes[folder] = suffixes
        print(f"Found {len(suffixes)} files in {folder} with prefix '{folder_name}'")
    
    # Find common suffixes across all folders
    if folder_suffixes:
        common_suffixes = set.intersection(*folder_suffixes.values())
        print(f"Found {len(common_suffixes)} common suffixes: {sorted(common_suffixes)}")
        
        # Prune non-common files from all folders
        for folder in input_folders:
            folder_name = os.path.basename(folder.rstrip(os.sep))
            print(f"Pruning folder: {folder}")
            prune_folder(folder, folder_name, common_suffixes)
    else:
        print("No valid folders found")

if __name__ == "__main__":
    main()
