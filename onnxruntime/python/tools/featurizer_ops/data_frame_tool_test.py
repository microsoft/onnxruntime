#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import argparse
import onnxruntime as onnxrt
import numpy as np
import pandas as pd
from data_frame_tool import DataFrameTool
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Test Feed Inputs utility')
    parser.add_argument('model_path', help='model path')
    parser.add_argument('-profile', action='store_true', help='enable chrome timeline trace profiling.')
    args = parser.parse_args()

    # Create options and the tool
    sess_options = onnxrt.SessionOptions()
    sess_options.enable_profiling = args.profile

    df_tool = DataFrameTool(args.model_path, sess_options)

    # Create a DataFrame that holds 3 inputs, string, bool, float in their respective columns
    df = pd.DataFrame([['string_input', 3.25, 8, 16, 32, 64, True, 0.25]],
                      columns=[
                          'StringInput', 'DoubleInput', 'Int8Input', 'Int16Input', 'Int32Input', 'Int64Input',
                          'BoolInput', 'Float32Input'
                      ])

    outputs = df_tool.execute(df, [])
    print('Outputs: ', outputs)


if __name__ == "__main__":
    sys.exit(main())
