import sys
from onnx import *
from onnxruntime import *
import numpy as np

def create_proxy_model(model_name, input_names, input_types, input_shapes, output_names, output_types, output_shapes):

    input_tensor_shapes = []
    for i, input_name in enumerate(input_names):
        input_tensor_shapes.append(helper.make_tensor_value_info(input_name, input_types[i], input_shapes[i]))
    
    output_tensor_shapes = []
    for i, output_name in enumerate(output_names):
        output_tensor_shapes.append(helper.make_tensor_value_info(output_name, output_types[i], output_shapes[i]))
        
    remote = helper.make_node('RemoteCall', input_names, output_names, domain='com.microsoft')
    graph = helper.make_graph([remote], 'graph', input_tensor_shapes, output_tensor_shapes)
    model = onnx.helper.make_model(graph)
    save(model, model_name)
    
if __name__ == "__main__":
    #todo - sanity check on inputs 
    if len(sys.argv) != 8:
        print ("Invalid input format! Sample inputs are: \n abc.onnx, [['x']], [TensorProto.FLOAT], [[-1]], [['y']], [TensorProto.FLOAT], [[-1]]")
        sys.exit()
    model_name = sys.argv[1]
    input_names = sys.argv[2]
    input_types = sys.argv[3]
    input_shapes = sys.argv[4]
    output_names = sys.argv[5]
    output_types = sys.argv[6]
    output_shapes = sys.argv[7]
    create_proxy_model(model_name, input_names, input_types, input_shapes, output_names, output_types, output_shapes)