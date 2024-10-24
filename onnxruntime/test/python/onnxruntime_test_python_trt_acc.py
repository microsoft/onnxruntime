"""
This test compares output of below huggingface models
- microsoft/resnet-18 and microsoft/resnet-50
- microsoft/Phi-3.5-mini-instruct with 1 layer transformer vs full model
on Pytorch cpu vs [ORT CPU EP, ORT TensorRT EP] with different configuations [fp16, no ort graph optimization]).
"""
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from transformers.onnx import export
import onnxruntime as ort
import numpy as np
import time
import unittest
import onnx
from onnx import helper, TensorProto

def run_model_in_pytorch(model, inputs):
    with torch.no_grad():
        output = model(**inputs).logits
    return output

def run_model_in_ort(model_file, inputs, ep, use_graph_opt=True):
    if use_graph_opt:
        sess_opt = None
    else:
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model_file, providers=ep, sess_opt=sess_opt)
    outputs = session.run(None, inputs)
    output = np.array(outputs[0])
    return output


def get_model_and_inputs(model_name, use_minimal_model=True):
    if model_name == "microsoft/resnet-50" or model_name == "microsoft/resnet-18":
        model = ResNetForImageClassification.from_pretrained(model_name)
        input_tensor = torch.randn(1, 3, 224, 224)
        pytorch_inputs = {'pixel_values': input_tensor}
        # inputs key value need to match forward()
        ort_inputs = {'pixel_values': input_tensor.numpy()}
    elif model_name == "microsoft/Phi-3.5-mini-instruct":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_minimal_model:
            print(f"Using 1 layer model for {model_name}")
            # Reduce the number of hidden layers (for example, keeping only 1 layer)
            model.model.layers = model.model.layers[:1]
            # Update the configuration to reflect the reduced number of layers
            model.config.num_hidden_layers = 1 # default 32
        else:
            print(f"Using full model for {model_name}")
            # model.model.layers = model.model.layers[:4]
            # # Update the configuration to reflect the reduced number of layers
            # model.config.num_hidden_layers = 4 # default 32
        dim = (1, 30)
        input_ids = torch.randint(0, 32064, dim)  # 32064 is vocab size
        attention_masks = torch.ones(*dim, dtype=torch.int64)

        # Prepare the inputs for the model
        pytorch_inputs = {'input_ids': input_ids, 'attention_mask':attention_masks}
        # inputs key value need to match forward()
        ort_inputs = {
            'input_ids': pytorch_inputs['input_ids'].numpy(),
            'attention_mask': pytorch_inputs['attention_mask'].numpy(),
            'onnx::Neg_2': torch.ones(1, dtype=torch.int64).numpy() # ORT requires this input since it's in the exported graph
        }
    return model, pytorch_inputs, ort_inputs

def get_ep(use_tensorrt=True, use_fp16=True):
    if not use_tensorrt:
        return [('CPUExecutionProvider', {})]
    else:
        return [
            ('TensorrtExecutionProvider', {'trt_fp16_enable': use_fp16})
            ]

"""
This hacky fix is required to fix onnx model graph. Github issue: https://github.com/pytorch/pytorch/issues/138637
Some slice nodes are missing starts/end attributes after onnx.export
"""
def fix_phi35_model(onnx_model_filename):
    model = onnx.load(onnx_model_filename)
    graph = model.graph

    # Iterate through nodes to find the node by name
    for node in graph.node:
        if node.name == "/model/layers.0/mlp/Slice_1":
            node.input[1] = "/model/layers.0/mlp/Constant_6_output_0" # starts attribute
            node.input[2] = "/model/layers.0/mlp/Constant_7_output_0" # ends attribute

        if node.name == "/model/layers.0/mlp/Slice":
            node.input[2] = "/model/layers.0/mlp/Constant_6_output_0" # ends attribute

        if node.name == "/Slice":
            node.input[1] = "/Constant41_output_0" # ends attribute

    # /model/layers.0/mlp/Slice_1 starts and /model/layers.0/mlp/Slice ends should be [8192]
    constant_tensor = helper.make_tensor(
        name="value",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=b'\x00 \x00\x00\x00\x00\x00\x00', # Binary of 8192
        raw=True
    )
    constant_node = helper.make_node(
        op_type="Constant",
        inputs=[],                               # No inputs for a Constant node
        outputs=["/model/layers.0/mlp/Constant_6_output_0"],
        name="/model/layers.0/mlp/Constant_6",
        value=constant_tensor
    )
    model.graph.node.append(constant_node)

    # /model/layers.0/mlp/Slice_1 attribute ends should be [16384]
    constant_tensor = helper.make_tensor(
        name="value",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=b'\x00@\x00\x00\x00\x00\x00\x00', # Binary of 16384
        raw=True
    )
    constant_node = helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=["/model/layers.0/mlp/Constant_7_output_0"],
        name="/model/layers.0/mlp/Constant_7",
        value=constant_tensor
    )
    model.graph.node.append(constant_node)

    # /Slice starts attr should be 0
    constant_tensor = helper.make_tensor(
        name="value",
        data_type=TensorProto.INT64,  
        dims=[1],                    
        vals=b'\x00\x00\x00\x00\x00\x00\x00\x00',
        raw=True
    )
    constant_node = helper.make_node(
        op_type="Constant",
        inputs=[],                               # No inputs for a Constant node
        outputs=["/Constant41_output_0"],
        name="/Constant41",
        value=constant_tensor
    )
    model.graph.node.append(constant_node)

    # Overwrite old model file with external weights since Phi3.5 full model exeeds 2GB
    onnx.save_model(model, onnx_model_filename, save_as_external_data=True, all_tensors_to_one_file=True, location="external_weights", size_threshold=1024, convert_attribute=False)

def run_comparison(self, model_name, use_minimal_model=True, use_tensorrt=True, use_fp16=True, use_graph_opt=True, rtol=1e-2, atol=1e-2):
    start_time = time.time()
    model, pytorch_inputs, ort_inputs = get_model_and_inputs(model_name, use_minimal_model)
    pytorch_output = run_model_in_pytorch(model, pytorch_inputs)
    pytorch_output = pytorch_output.numpy()
    suffix = "_min" if use_minimal_model else ""
    model_file = model_name.split("/")[1] + suffix + ".onnx"
    # Export pytorch model to onnx
    input_names = list(pytorch_inputs.keys())
    torch.onnx.export(model, pytorch_inputs, model_file, input_names=input_names)
    if model_name == "microsoft/Phi-3.5-mini-instruct":
        fix_phi35_model(model_file)
    providers = get_ep(use_tensorrt, use_fp16)
    ort_output = run_model_in_ort(model_file, ort_inputs, providers, use_graph_opt=use_graph_opt)
    # print(f"pytorch_output={pytorch_output}")
    # print(f"ort_output={ort_output}")
    are_close = np.allclose(pytorch_output, ort_output, rtol=rtol, atol=atol)
    # print(f"====\n{model_name}{suffix} [FP16={use_fp16} use_graph_opt={use_graph_opt}] pytorch CPU and ORT {providers[0][0]} results are allclose with atol={atol} and rtol={rtol}")
    self.assertTrue(are_close, f"====\n{model_name}{suffix} FP16={use_fp16} "  \
        "use_graph_opt={use_graph_opt} pytorch CPU and ORT {providers[0][0]} results " \
        "should be close with atol={atol} and rtol={rtol}")
    difference = np.linalg.norm(ort_output - pytorch_output)
    print("Difference:", difference)
    diff = np.abs(ort_output - pytorch_output).mean()
    print(f"Mean absolute difference: {diff}")
    rel_diff = np.abs(ort_output - pytorch_output) / np.abs(pytorch_output + 1e-8)  # Add epsilon to avoid division by zero
    print(f"Max relative difference: {np.max(rel_diff)}")
    end_time = time.time()
    print(f"Time : {end_time - start_time:.6f} seconds")

"""
Test Resnet18 and Resnet50 with different configurations
"""
class TestResnetAccuracy(unittest.TestCase):
    # We currently only test CUDA/TRT EP due to users only raise this issue when using CUDA/TRT EP.
    @unittest.skipIf(
        "TensorrtExecutionProvider" not in ort.get_available_providers()
        and "CUDAExecutionProvider" not in ort.get_available_providers(),
        reason="Test CUDA/TRT EP only",
    )

    def test_resnet18_cpu_fp32_wo_opt(self):
        run_comparison(self, "microsoft/resnet-18", 
            use_minimal_model=False, use_tensorrt=False, use_fp16=False, use_graph_opt=False)
    
    def test_resnet18_cpu_fp32(self):
        run_comparison(self, "microsoft/resnet-18", 
            use_minimal_model=False, use_tensorrt=False, use_fp16=False, use_graph_opt=True)

    def test_resnet18_cpu_fp32(self):
        run_comparison(self, "microsoft/resnet-18", 
            use_minimal_model=False, use_tensorrt=True, use_fp16=False, use_graph_opt=True)

    def test_resnet18_trt_fp32(self):
        run_comparison(self, "microsoft/resnet-18", 
            use_minimal_model=False, use_tensorrt=True, use_fp16=True, use_graph_opt=True)

    def test_resnet18_trt_fp16(self):
        run_comparison(self, "microsoft/resnet-18", 
            use_minimal_model=False, use_tensorrt=True, use_fp16=False, use_graph_opt=True)

    def test_resnet50_trt_fp16(self):
        run_comparison(self, "microsoft/resnet-50", 
            use_minimal_model=False, use_tensorrt=True, use_fp16=False, use_graph_opt=True)

"""
Test Phi3.5 (1 layer) and full Phi3.5 with different configurations
"""
class TestPhi35Accuracy(unittest.TestCase):
    # We currently only test CUDA/TRT EP due to users only raise this issue when using CUDA/TRT EP.
    @unittest.skipIf(
        "TensorrtExecutionProvider" not in ort.get_available_providers()
        and "CUDAExecutionProvider" not in ort.get_available_providers(),
        reason="Test CUDA/TRT EP only",
    )

    def test_phi35_1l_cpu_fp32_wo_opt(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", 
            use_minimal_model=True, use_tensorrt=False, use_fp16=False, use_graph_opt=False)
    
    def test_phi35_1l_cpu_fp32(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", 
            use_minimal_model=True, use_tensorrt=False, use_fp16=False, use_graph_opt=True)

    def test_phi35_1l_trt_fp32(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", 
            use_minimal_model=True, use_tensorrt=True, use_fp16=False, use_graph_opt=True)

    def test_phi35_1l_trt_fp16(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", 
            use_minimal_model=True, use_tensorrt=True, use_fp16=True, use_graph_opt=True,
            rtol=1e-1, atol=1e-1) # Need to relax rtol and atol for fp16 test case to pass

    def test_phi35_full_trt_fp16(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", 
            use_minimal_model=False, use_tensorrt=True, use_fp16=True, use_graph_opt=True)


if __name__ == "__main__":
    unittest.main()
