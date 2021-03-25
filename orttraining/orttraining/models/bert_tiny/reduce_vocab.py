import sys
import onnx
from onnx import helper, shape_inference
from onnx import TensorProto
import numpy as np
from onnx import numpy_helper

if len(sys.argv) < 2:
    print("Please give model path...")
    exit(1)

input_model_name = sys.argv[1]
output_model_name = input_model_name[:-5] + '_reduce_vocab.onnx'

model = onnx.load(input_model_name)

new_initializer = []
for initializer in model.graph.initializer:
    if initializer.name == 'bert.embeddings.word_embeddings.weight':
        w = numpy_helper.to_array(initializer)
        new_w = numpy_helper.from_array(w[0:128,:], 'bert.embeddings.word_embeddings.weight');
        new_initializer.append(new_w)
    elif initializer.name == 'cls.predictions.bias':
        w = numpy_helper.to_array(initializer)
        new_w = numpy_helper.from_array(w[0:128], 'cls.predictions.bias');
        new_initializer.append(new_w)
    else:
        new_initializer.append(initializer)
    
del model.graph.initializer[:]
model.graph.initializer.extend(new_initializer)

del model.graph.output[0].type.tensor_type.shape.dim[-1]
dim = model.graph.output[0].type.tensor_type.shape.dim.add()
dim.dim_value = 128

f = open(output_model_name, "wb")
f.write(model.SerializeToString())
f.close()
