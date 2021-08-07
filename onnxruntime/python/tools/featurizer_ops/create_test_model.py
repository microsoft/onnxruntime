#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import onnx
import numpy as np
import os
import sys
import argparse
from onnx import numpy_helper
from onnx import helper
from onnx import utils
from onnx import AttributeProto, TensorProto, GraphProto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, help="Model file name to save")
    return parser.parse_args()


def create_model():
    """
     This function creates a test feed model that consists of a single node that takes
     Tensors of all inputs
     """
    args = parse_arguments()

    # bool_identity
    bool_input = helper.make_tensor_value_info('BoolInput', TensorProto.BOOL, [1, 1])
    # Create output for Identity
    bool_output = helper.make_tensor_value_info('BoolOutput', TensorProto.BOOL, [1, 1])
    # Create node def
    bool_identity_def = helper.make_node('Identity', inputs=['BoolInput'], outputs=['BoolOutput'], name='BoolIdentity')

    # Create string_identity
    string_input = helper.make_tensor_value_info('StringInput', TensorProto.STRING, [1, 1])
    string_output = helper.make_tensor_value_info('StringOutput', TensorProto.STRING, [1, 1])
    string_identity_def = helper.make_node('Identity',
                                           inputs=['StringInput'],
                                           outputs=['StringOutput'],
                                           name='StringIdentity')

    # double
    double_input = helper.make_tensor_value_info('DoubleInput', TensorProto.DOUBLE, [1, 1])
    double_output = helper.make_tensor_value_info('DoubleOutput', TensorProto.DOUBLE, [1, 1])
    double_identity_def = helper.make_node('Identity',
                                           inputs=['DoubleInput'],
                                           outputs=['DoubleOutput'],
                                           name='DoubleIdentity')

    # int8
    int8_input = helper.make_tensor_value_info('Int8Input', TensorProto.INT8, [1, 1])
    int8_output = helper.make_tensor_value_info('Int8Output', TensorProto.INT8, [1, 1])
    int8_identity_def = helper.make_node('Identity', inputs=['Int8Input'], outputs=['Int8Output'], name='Int8Identity')

    # int16
    int16_input = helper.make_tensor_value_info('Int16Input', TensorProto.INT16, [1, 1])
    int16_output = helper.make_tensor_value_info('Int16Output', TensorProto.INT16, [1, 1])
    int16_identity_def = helper.make_node('Identity',
                                          inputs=['Int16Input'],
                                          outputs=['Int16Output'],
                                          name='Int16Identity')

    # int32
    int32_input = helper.make_tensor_value_info('Int32Input', TensorProto.INT32, [1, 1])
    int32_output = helper.make_tensor_value_info('Int32Output', TensorProto.INT32, [1, 1])
    int32_identity_def = helper.make_node('Identity',
                                          inputs=['Int32Input'],
                                          outputs=['Int32Output'],
                                          name='Int32Identity')

    # int64
    int64_input = helper.make_tensor_value_info('Int64Input', TensorProto.INT64, [1, 1])
    int64_output = helper.make_tensor_value_info('Int64Output', TensorProto.INT64, [1, 1])
    int64_identity_def = helper.make_node('Identity',
                                          inputs=['Int64Input'],
                                          outputs=['Int64Output'],
                                          name='Int64Identity')

    ##### Optional input as it has initializer. This one is interesting bc it needs float32 which
    # Pandas do not have
    # Create Initializer with optional input with default value from the initializer
    float32_input = helper.make_tensor_value_info('Float32Input', TensorProto.FLOAT, [1, 1])
    float32_output = helper.make_tensor_value_info('Float32Output', TensorProto.FLOAT, [1, 1])
    optional_identity_def = helper.make_node('Identity',
                                             inputs=['Float32Input'],
                                             outputs=['Float32Output'],
                                             name='OptionalIdentity')

    # Create a default initializer for float32_input.
    tensor_float32 = helper.make_tensor(name='Float32Input',
                                        data_type=TensorProto.FLOAT,
                                        dims=[1, 1],
                                        vals=np.array([[.0]]).astype(np.float32),
                                        raw=False)

    # Make a graph
    graph_def = helper.make_graph(nodes=[
        bool_identity_def, string_identity_def, double_identity_def, int8_identity_def, int16_identity_def,
        int32_identity_def, int64_identity_def, optional_identity_def
    ],
                                  name='optional_input_graph',
                                  inputs=[
                                      bool_input, string_input, double_input, int8_input, int16_input, int32_input,
                                      int64_input, float32_input
                                  ],
                                  outputs=[
                                      bool_output, string_output, double_output, int8_output, int16_output,
                                      int32_output, int64_output, float32_output
                                  ],
                                  initializer=[tensor_float32])

    model_def = helper.make_model(graph_def, producer_name='feed_inputs_test')
    onnx.checker.check_model(model_def)
    onnx.helper.strip_doc_string(model_def)
    final_model = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(final_model)
    onnx.save(final_model, args.output_file)


if __name__ == "__main__":
    sys.exit(create_model())
