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

# Originally from process_images_with_debug_op.bat
def preprocess_images_with_debug_op(model_file, image_files):
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

    data = []  # temp empty list to collect the data
    spaces_removed_data = [] # temp empty list to collect float outputs 
    scale = 0.05
    zero_point = 0
    logs_directory = os.getcwd() # log files location  # TODO -- Make this an argument?
    # image_directory = os.path.join(logs_directory , "images") # dataset images location
    maxvalue = {} # dictionary to store max values
    minvalue = {} # dictionary to store min values

    # loop through all the images in a directory
    for image in os.listdir(image_directory):
        image_file = os.path.join(image_directory, image)

        #call the batch file to process the image[i] in the location and generate output log files
        # --> TODO <--
        preprocess_images_with_debug_op(image_file)
    
        # open each log file and read through it line by line and ignore 'labels' file 
        for filename in os.listdir(logs_directory):
            if filename.endswith(".txt") and (filename.startswith("labels.txt") == False): 
                with open(os.path.join(logs_directory, filename), 'r') as file_object:
                    line = file_object.readline()
                    spaces_removed_data = []
                    while line:
                        # at each line check for a match with a regex
                        key, match = _parse_line(line)
                        if key == 'data': 
                            data = match.group(0)
                            for i in data.split(','):
                                try:
                                    if i != '':
                                        spaces_removed_data.append(float(i.strip()))
                                except ValueError:
                                    print("error on line"+ i)

                            #add min and max values to the dictionary
                            if (filename.split(".")[0] not in maxvalue.keys()) or (max(spaces_removed_data) > maxvalue[filename.split(".")[0]]):
                                maxvalue[filename.split(".")[0]] = max(spaces_removed_data)
                            if (filename.split(".")[0] not in minvalue.keys()) or (min(spaces_removed_data) < minvalue[filename.split(".")[0]]):
                                minvalue[filename.split(".")[0]] = min(spaces_removed_data)
                    
                        line = file_object.readline() #next line in file

                    file_object.close() #close the file object
    
    #clean up the log files to repeat steps for the next image
    cleanup_images_from_debug_op_run()

    #print(json.dumps(maxvalue, indent = 4)) #dump the max dictionary
    #print(json.dumps(minvalue, indent = 4)) #dump the min dictionary
    for conv_output in maxvalue:
        scale = (maxvalue[conv_output] - minvalue[conv_output]) / 255 if minvalue[conv_output] !=maxvalue[conv_output] else 1
        zero_point = round((0 - minvalue[conv_output]) / scale)
        print(f"{conv_output} : {scale} , {zero_point}")

        # Replace by this
        # quantized_model = quantize(model, per_channel=False, nbits=8, quantization_mode=QuantizationMode.QLinearOps_Dynamic,
# 		asymmetric_input_types=False, input_quantization_params=_input_quantization_params,
# 		output_quantization_params=_output_quantization_params)
