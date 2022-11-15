import onnx
from onnx import helper, shape_inference

model = onnx.load('bert_base_cased_1_fp16_gpu.onnx')

for input in model.graph.input:
    new_shape = []
    for dim in input.type.tensor_type.shape.dim:
        if dim.dim_param == "batch_size":
            new_shape.append(1)
        elif dim.dim_param == "seq_len":
            new_shape.append(128)
        else:
            new_shape.append(dim.dim_value)
    del input.type.tensor_type.shape.dim[:]
    for shape in new_shape:
        dim = input.type.tensor_type.shape.dim.add()
        dim.dim_value = shape

for output in model.graph.output:
    new_shape = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.dim_param == "batch_size":
            new_shape.append(1)
        elif dim.dim_param == "seq_len":
            new_shape.append(128)
        else:
            new_shape.append(dim.dim_value)
    del output.type.tensor_type.shape.dim[:]
    for shape in new_shape:
        dim = output.type.tensor_type.shape.dim.add()
        dim.dim_value = shape

for value_info in model.graph.value_info:
    new_shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param == "batch_size":
            new_shape.append(1)
        elif dim.dim_param == "seq_len":
            new_shape.append(128)
        else:
            new_shape.append(dim.dim_value)
    del value_info.type.tensor_type.shape.dim[:]
    for shape in new_shape:
        dim = value_info.type.tensor_type.shape.dim.add()
        dim.dim_value = shape

onnx.save(model, 'bert_base_cased_1_fp16_gpu_shaped.onnx')