from onnx_model import OnnxModel
import onnx
from fusion_utils import NumpyHelper

model_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small.onnx"
model = OnnxModel(onnx.load(model_path))
for node in model.nodes():
    if "model_modeling_mixformer_sequential_ParallelBlock_sub1" in node.name:
        for input in node.input:
            print(input)
            tensor = model.get_initializer(input)
            if tensor is None:
                print("None")
            else:
                np_array = NumpyHelper.to_array(tensor)
                print(np_array.shape)
                print(np_array)
    print(node.name)
