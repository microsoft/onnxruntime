#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import argparse
import onnxruntime as onnxrt
import numpy as np
import pandas as pd
from feed_inputs import DataFrameTool
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Test Feed Inputs utility')
    parser.add_argument('model_path', help='model path')
    parser.add_argument('-debug', action='store_true',
                        help='pause execution to allow attaching a debugger.')
    parser.add_argument('-profile', action='store_true',
                        help='enable chrome timeline trace profiling.')
    args = parser.parse_args()

    # Load model
    sess_options = onnxrt.SessionOptions()
    sess_options.enable_profiling = args.profile

    sess = onnxrt.InferenceSession(args.model_path, sess_options)

    # Create a DataFrame that holds 3 inputs, string, bool, float in their respective columns
    df = pd.DataFrame([['string_input', True, np.float32(0.25)]], index=[0], columns=['F2', 'Label', 'F1'])

    feed_helper = DataFrameTool(sess)
    feeds = feed_helper.feed_nputs(df)
    sess.run([], feeds)
    print('Run complete')

if __name__ == "__main__":
    sys.exit(main())
