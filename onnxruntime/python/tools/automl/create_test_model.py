# Create a test model and then try to run it
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

# This function creates a test feed model that consists of a single node that takes
# Tensors of all inputs
def create_model():
     args = parse_arguments()

     # Create Identity node
     # Create input for Identity
     label_input = helper.make_tensor_value_info('Label', TensorProto.BOOL, [1,1])
     # Create output for Identity
     label0_output = helper.make_tensor_value_info('Label0', TensorProto.BOOL, [1,1])
     # Create Identity node
     Identity_def = helper.make_node('Identity', inputs=['Label'], outputs=['Label0'], name='Identity')

     # Create Identity1
     f2_input = helper.make_tensor_value_info('F2', TensorProto.STRING, [1,1])
     f20_output = helper.make_tensor_value_info('F20', TensorProto.STRING, [1,1])
     Identity1_def = helper.make_node('Identity', inputs=['F2'], outputs=['F20'], name='Identity1')

     # Create Identity0 with optional input with default value from the initializer
     f1_input = helper.make_tensor_value_info('F1', TensorProto.FLOAT, [1,1])
     f11_output = helper.make_tensor_value_info('F11', TensorProto.FLOAT, [1,1])
     Identity0_def = helper.make_node('Identity', inputs=['F1'], outputs=['F11'], name='Identity0')

     # Create a default initializer for F1 input
     tensor_f1 = helper.make_tensor(name='F1', data_type=TensorProto.FLOAT, dims=[1,1],
                                        vals=np.array([[.0]]).astype(np.float32), raw=False)

     # Make a graph
     graph_def = helper.make_graph(nodes=[Identity_def, Identity1_def, Identity0_def], name='optional_input_graph',
                                   inputs=[label_input, f2_input, f1_input], outputs=[label0_output, f20_output, f11_output],
                                   initializer=[tensor_f1])

     model_def = helper.make_model(graph_def, producer_name='feed_inputs_test')
     final_model = onnx.utils.polish_model(model_def)
     onnx.save(final_model, args.output_file)

if __name__ == "__main__":
    sys.exit(create_model())


