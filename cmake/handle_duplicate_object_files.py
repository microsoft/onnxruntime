#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import shutil


# Note: This script is mainly used for handling extracting duplicate named .o files under different subdirectories for
# each onnxruntime library. (Only applicable when doing a Mac Catalyst build.)
def main():
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    for subdir, dirs, files in os.walk(source_dir):
        for file_name in files:
            if file_name.endswith(".o"):
                dest_name_without_extension, _ = os.path.splitext(file_name)
                counter = 0

                dest_file = f"{dest_name_without_extension}.o"
                while os.path.exists(os.path.join(dest_dir, dest_file)):
                    print("Duplicates" + os.path.join(dest_dir, dest_file))
                    counter += 1
                    dest_file = f"{dest_name_without_extension}_{counter}.o"

                destination_path = os.path.join(dest_dir, dest_file)
                source_file = os.path.join(source_dir, subdir, file_name)
                shutil.copy(source_file, destination_path)


if __name__ == "__main__":
    main()
