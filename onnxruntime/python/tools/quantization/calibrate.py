#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import re
import subprocess
import json

import onnx

# Originally from process_images_with_debug_op.bat
def preprocess_images_with_debug_op(model_file, image_files):
    """
    Carries out inference in an instrumented model that can log
    outputs of quantizable operators *or* using an unmodified model
    but through a suitably instrumented inference engine that can 
    log outputs of quantizable operators.
    """
    for img in image_files:
        # FIXME: Can we do this asynchronously?
        subprocess.call([r'debug-operator-sample.exe', model_file, img])

def cleanup_images_from_debug_op_run():
    # TODO -- implement cleanup phase
    pass

def calibrate_loop(image_directory):
    """
    Loops through the images in `image_directory` and collects
    distribution of outputs on relevant nodes for each inference.
    """
    logs_directory = os.getcwd() # log files location  # TODO -- Make this an argument?
    # image_directory = os.path.join(logs_directory , "images") # dataset images location
    # loop through all the images in a directory
    for image in os.listdir(image_directory):
        image_file = os.path.join(image_directory, image)

        #call the batch file to process the image[i] in the location and generate output log files
        # --> TODO <--
        preprocess_images_with_debug_op(image_file)

    sfacs, zpts = process_logfiles(logs_directory)
    # quantized_model = quantize(model, per_channel=False, nbits=8, quantization_mode=QuantizationMode.QLinearOps_Dynamic,
# 		asymmetric_input_types=False, input_quantization_params=sfacs,
# 		output_quantization_params=zpts)

def process_logfiles(logs_directory):
    """
    Open each log file in :param logs_directory: and read through it line
    by line and ignore 'labels' file
    """
    rx_dict = {
        'dimensions': re.compile(r'dimensions: (.*)\n'),
        'data type': re.compile(r'data type: (\d+)\n'),
        'data': re.compile(r'(?<=data:).*'),
    }

    def _parse_line(line):
        """
        Do a regex search against all defined regexes and
        return the key and match result of the first matching regex
        """
        for key, rx in rx_dict.items():
            match = rx.search(line)
            if match:
                return key, match
        # if there are no matches
        return None, None

    data = []  # temp empty list to collect the data
    spaces_removed_data = [] # temp empty list to collect float outputs 
    scale = 0.05
    zero_point = 0

    maxvalue = {} # dictionary to store max values
    minvalue = {} # dictionary to store min values

    if not(os.path.exists(logs_directory)):
        raise ValueError(f"Log directory ${logs_directory} not found!")

    for filename in os.listdir(logs_directory):
        if (not filename.endswith(".txt")) or filename.startswith("labels.txt"):
            next

        # FIXME -- maybe log this
        # print(f"Processing: {filename}.")
                                               
        with open(os.path.join(logs_directory, filename), 'r') as file_object:
            spaces_removed_data = []
            for line in file_object:
                # at each line check for a match with a regex
                key, match = _parse_line(line)
                if key != 'data':
                    continue
                data = match.group(0)
                for i in data.split(','):
                    try:
                        if i != '':
                            spaces_removed_data.append(float(i.strip()))
                    except ValueError:
                        pass  # Ignore errors

                # Add min and max values to the dictionary
                fname_key = filename.split(".")[0]
                if (fname_key not in maxvalue) or (max(spaces_removed_data) > maxvalue[fname_key]):
                    maxvalue[fname_key] = max(spaces_removed_data)
                if (fname_key not in minvalue) or (min(spaces_removed_data) < minvalue[filename.split(".")[0]]):
                    minvalue[fname_key] = min(spaces_removed_data)

    #clean up the log files to repeat steps for the next image
    cleanup_images_from_debug_op_run()

    scalefacs = {}
    zpoints = {}

    for conv_output in maxvalue:
        scalefacs[conv_output] = (maxvalue[conv_output] - minvalue[conv_output]) / 255 if minvalue[conv_output] !=maxvalue[conv_output] else 1
        zpoints[conv_output] = round((0 - minvalue[conv_output]) / scale)

    return (scalefacs, zpoints)

def main():
    # calibrate_loop()
    pass

if __name__ == '__main__':
    # parse command-line arguments with argparse
    main()
