# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import onnx
import os
import sys
 
from onnx.onnx_pb import ModelProto
from google.protobuf import text_format
 
onnx_operator_list = [
    "Abs", "Acos", "Add", "And", "ArgMax", "ArgMin", "Asin", "Atan", "AveragePool", "BatchNormalization",
    "Cast", "Ceil", "Clip", "Concat", "Constant", "Conv", "ConvTranspose", "Cos", "DepthToSpace", "Div",
    "Dropout", "Elu", "Equal", "Exp", "Flatten", "Floor", "GRU", "Gather", "Gemm", "GlobalAveragePool",
    "GlobalLpPool", "GlobalMaxPool", "Greater", "HardSigmoid", "Hardmax", "Identity",
    "InstanceNormalization", "LRN", "LSTM", "LeakyRelu", "Less", "Log", "LogSoftmax", "LpNormalization",
    "LpPool", "MatMul", "Max", "MaxPool", "MaxRoiPool", "Mean", "Min", "Mul", "Multinomial", "Neg", "Not",
    "Or", "PRelu", "Pad", "Pow", "RNN", "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
    "Reciprocal", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", 
    "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape", "Selu", "Shape", "Sigmoid",
    "Sin", "Size", "Slice", "Softmax", "Softplus", "Softsign", "SpaceToDepth", "Split", "Sqrt", "Squeeze",
    "Sub", "Sum", "Tan", "Tanh", "Tile", "TopK", "Transpose", "Unsqueeze", "Upsample", "Xor",
    "ATen", "Affine", "ConstantFill", "Crop", "GRUUnit", "GivenTensorFill", "If", "ImageScaler", "Loop",
    "LoopIndexTensor", "MeanVarianceNormalization", "ParametricSoftplus", "Scale", "ScaledTanh", "ThresholdedRelu"
]

#Given a model directory, it parses the model operators and outputs in CSV formst as below:
#        Abs, Acos, Add, Add, ...
# Model1,  5,   18,    ,   1, ...
# Model2,   ,  19,   3,     , ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Operators of ONNX model.')
    parser.add_argument('modelDir', nargs=1, help='Model Directory Path')
    parser.add_argument('resultFile', nargs=1, help='Result File Path')
    
    model_dir = parser.parse_args().modelDir[0]
    result_file = parser.parse_args().resultFile[0]
    
    with open(result_file, 'a') as result:

        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.onnx'):
                    abs_path = root + "/" + file
                    f = open(abs_path, "rb")
                    proto = ModelProto()
                    proto.ParseFromString(f.read())
                    f.close()

                    # Get model operator list
                    operators = {}
                    nodes = proto.graph.node
                    for node in nodes:
                        print(node.op_type)
                        if not node.op_type in operators:
                          operators[node.op_type]=1
                        else:
                          operators[node.op_type]+=1

                    # store all the operators with format:
                    # ModelName[,occurences]* 
                    
                    #head
                    head = ''
                    for op in onnx_operator_list:
                       head += ',{}'.format(op)
                    result.write('{}\n'.format(head))

                    #content
                    model_name = file[:-5]
                    if model_name == 'model':
                        model_name = os.path.basename(os.path.dirname(abs_path)) # use directory name

                    model_operators_string = '{}'.format(model_name)
                    for op in onnx_operator_list:
                        model_operators_string+=','
                        if op in operators:
                            model_operators_string+='{}'.format(operators[op])
                    result.write('{}\n'.format(model_operators_string))
