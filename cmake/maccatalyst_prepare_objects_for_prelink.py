#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import sys


# Note: This script is mainly used for sanity checking/validating the files in the .a library equal to the .o files
# in the source dir to handle the case of source files having duplicate names under different subdirectories for
# each onnxruntime library. (Only applicable when doing a Mac Catalyst build.)
def main():
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    files_from_static_lib = sys.argv[3]
    files_from_source_dir = []
    for subdir, _, files in os.walk(source_dir):
        for file_name in files:
            if file_name.endswith(".o"):
                files_from_source_dir.append(file_name.strip())
                dest_name_without_extension, _ = os.path.splitext(file_name)
                counter = 0

                dest_file = f"{dest_name_without_extension}.o"
                while os.path.exists(os.path.join(dest_dir, dest_file)):
                    print("Duplicate file name from source: " + os.path.join(source_dir, subdir, file_name))
                    counter += 1
                    dest_file = f"{dest_name_without_extension}_{counter}.o"
                    print("Renamed file name in destination: " + os.path.join(dest_dir, dest_file))

                destination_path = os.path.join(dest_dir, dest_file)
                source_file = os.path.join(source_dir, subdir, file_name)
                shutil.copy(source_file, destination_path)

    # Sanity check to ensure the number of .o object from the original cmake source directory matches with the number
    # of .o files extracted from each .a onnxruntime library
    file_lists_from_static_lib = []
    with open(files_from_static_lib) as file:
        filenames = file.readlines()
    for filename in filenames:
        file_lists_from_static_lib.append(filename.strip())

    sorted_list1 = sorted(file_lists_from_static_lib)
    sorted_list2 = sorted(files_from_source_dir)

    if len(sorted_list1) != len(sorted_list2):
        print(
            "Caught a mismatch in the number of .o object files from the original cmake source directory: ",
            len(sorted_list1),
            "the number of .o files extracted from the static onnxruntime lib: ",
            len(sorted_list2),
            "for: ",
            os.path.basename(source_dir),
        )

    if sorted_list1 == sorted_list2:
        print(
            "Sanity check passed: object files from original source directory matches with files extracted "
            "from static library for: ",
            os.path.basename(source_dir),
        )
    else:
        print(
            "Error: Mismatch between object files from original source directory "
            "and the .o files extracted from static library for: ",
            os.path.basename(source_dir),
        )


if __name__ == "__main__":
    main()
