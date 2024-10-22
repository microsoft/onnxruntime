"""
This test compares output of below huggingface models
- "microsoft/resnet-50"
- "microsoft/Phi-3.5-mini-instruct"
on Pytorch cpu vs [ORT CPU EP, ORT TensorRT EP] with different configuations (fp16, no ort graph optimization, 1 layer transformer vs full model)
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

def run_model_in_ort(model_file, inputs, ep, disable_ort_graph_optimization=False):
    if disable_ort_graph_optimization:
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    else:
        sess_opt = None
    session = ort.InferenceSession(model_file, providers=ep, sess_opt=sess_opt)
    # model_inputs = session.get_inputs()
    # input_data = np.array(input_tensor) 
    # outputs = session.run(None, {model_inputs[0].name: input_data})
    outputs = session.run(None, inputs)
    output = np.array(outputs[0])
    return output


def get_model_and_inputs(model_name, use_minimal_model=True):
    if model_name == "microsoft/resnet-50":
        model = ResNetForImageClassification.from_pretrained(model_name)
        # if use_minimal_model:
        #     model.config.num_channels = 1 
        #     model.config.embedding_size = 1
        #     model.config.hidden_sizes = [1, 2]
        #     model.config.depths = [1, 2]
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
        # input_tensor = torch.randint(0, model.config.vocab_size, (1, 30))  # Batch size 1, sequence length 30
        # inputs = {'input_ids': random_input_ids}
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
This hacky fix is required to fix onnx model graph.
Some slice nodes are missing starts/end attributes after onnx.export
"""
def fix_phi35_model(onnx_model_filename):
    model = onnx.load(onnx_model_filename)
    graph = model.graph

    # Iterate through nodes to find the node by name
    for node in graph.node:
        if node.name == "/model/layers.0/mlp/Slice_1":
            # print(f"Found node: {node.name}")
            # print(node)  # Print the details of the node
            # print(node.input)
            node.input[1] = "/model/layers.0/mlp/Constant_6_output_0" # starts
            node.input[2] = "/model/layers.0/mlp/Constant_7_output_0" # ends

        if node.name == "/model/layers.0/mlp/Slice":
            # print(f"Found node: {node.name}")
            # print(node)  # Print the details of the node
            # print(node.input)
            node.input[2] = "/model/layers.0/mlp/Constant_6_output_0" # ends

        if node.name == "/Slice":
            # print(f"Found node: {node.name}")
            # print(node)  # Print the details of the node
            # print(node.input)
            node.input[1] = "/Constant41_output_0"
            # return
        # if node.name == "/model/layers.0/mlp/Mul_output_0":
        #     print(f"Found node: {node.name}")
        #     print(node)  # Print the details of the node
        #     # return
        # if node.name == "/model/layers.0/mlp/Constant_1_output_0":
        #     print(f"Found node: {node.name}")
        #     print(node)  # Print the details of the node
        # if node.name == "/model/layers.0/mlp/Mul_1":
        #     print(node)
        # if node.name == "/model/layers.0/mlp/Constant_1":
        #     print(node)
    
    # for initializer in graph.initializer:
    #     print(f"Name: {initializer.name}")
    #     tensor_value = onnx.numpy_helper.to_array(initializer)
    #     print(f"Value: {tensor_value}")
    #     print(tensor_value)
        # if initializer.name == "/model/layers.0/mlp/Mul_output_0":
        #     print(f"Tensor '{initializer.name}' found in initializers.")
        #     tensor_value = numpy_helper.to_array(initializer)
        #     print(f"Value: {tensor_value}")
        #     print(tensor_value)
        #     # return tensor_value
        # if initializer.name == "/model/layers.0/mlp/Constant_1_output_0":
        #     print(f"Tensor '{initializer.name}' found in initializers.")
        #     tensor_value = numpy_helper.to_array(initializer)
        #     print(f"Value: {tensor_value}")
        #     print(node)
    
    # for node in graph.output:
    #     print(node)
    #     if node.name == "/model/layers.0/mlp/Mul_output_0":
    #         print(f"Tensor '{node.name}' found (op_type: {node.op_type}) .")
    #         print(node)
    #         # return node
    #     if node.name == "/model/layers.0/mlp/Constant_1_output_0":
    #         print(f"Tensor '{node.name}' found (op_type: {node.op_type}) .")
    #         print(node)

    # for node in graph.node:
    #     if node.op_type == "Constant":
    #         print(node)

    # print(f"Node '{node_name}' not found in the model.")
    # data = np.array([8192], dtype=np.int64)
    # # raw_bytes = data.tobytes()
    # # # raw_bytes = struct('<q', 8192)
    # # print(raw_bytes)
    # /model/layers.0/mlp/Slice_1 starts and /model/layers.0/mlp/Slice ends 8192
    constant_tensor = helper.make_tensor(
        name="value",                # Attribute name
        data_type=TensorProto.INT64,  # Data type (7 = DOUBLE)
        dims=[1],                    # Dimensions (1 element)

        # vals=[8192], raw=False
        # vals=np.array([8192], dtype=np.int64).tobytes(), raw=True
        # vals=struct.pack('<q', 8192), raw=True
        # vals=b"\000\040\000\000\000\000\000\000", raw=True
        # vals=b"\000\040\000\000\000\000\000\000"
        # vals=np.array([8192]).flatten().astype(np.int64)
        # vals=b"\x00\x20\x00\x00\x00\x00\x00\x00", raw=True
        # vals=0x0000000000004000, raw=True
        # vals=b"\x30\x00\x00\x00\x00\x00\x00\x00", raw=True
        # vals=raw_bytes, raw=True
        vals=b'\x00 \x00\x00\x00\x00\x00\x00', raw=True
    )
    # # print(f"Created tensor={constant_tensor}")
    # # constant_tensor.raw_data=b'\x00\x20\x00\x00\x00\x00\x00\x00'
    # # print(f"Created tensor={constant_tensor}")
    # # print(f"raw_data type={type(constant_tensor.raw_data)}")
    constant_node = helper.make_node(
        op_type="Constant",                      # Operation type
        inputs=[],                               # No inputs for a Constant node
        outputs=["/model/layers.0/mlp/Constant_6_output_0"],  # Output name
        name="/model/layers.0/mlp/Constant_6",   # Node name
        value=constant_tensor                    # Attribute for constant value
    )
    model.graph.node.append(constant_node)
    # print(f"Created node ={constant_node}")

    # /model/layers.0/mlp/Slice_1 attribute ends
    constant_tensor = helper.make_tensor(
        name="value",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=b'\x00@\x00\x00\x00\x00\x00\x00', raw=True
    )
    constant_node = helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=["/model/layers.0/mlp/Constant_7_output_0"],
        name="/model/layers.0/mlp/Constant_7",
        value=constant_tensor
    )
    model.graph.node.append(constant_node)
    # /model/layers.0/mlp/Slice ends
    # constant_tensor = helper.make_tensor(
    #     name="value",
    #     data_type=TensorProto.INT64,
    #     dims=[1],
    #     # vals=[8192],
    #     # vals=np.array([8192]).flatten().astype(np.int64)
    #     # raw=False
    #     vals=b'\x00 \x00\x00\x00\x00\x00\x00', raw=True
    # )
    # constant_node = helper.make_node(
    #     op_type="Constant",                      # Operation type
    #     inputs=[],                               # No inputs for a Constant node
    #     outputs=["/model/layers.0/mlp/Constant_4_output_0"],  # Output name
    #     name="/model/layers.0/mlp/Constant_4",   # Node name
    #     value=constant_tensor                    # Attribute for constant value
    # )
    # model.graph.node.append(constant_node)
    # /Slice starts
    constant_tensor = helper.make_tensor(
        name="value",                # Attribute name
        data_type=TensorProto.INT64,  
        dims=[1],                    
        # vals=[8192],
        # vals=np.array([8192]).flatten().astype(np.int64)
        # raw=False
        vals=b'\x00\x00\x00\x00\x00\x00\x00\x00', raw=True
    )
    constant_node = helper.make_node(
        op_type="Constant",                      # Operation type
        inputs=[],                               # No inputs for a Constant node
        outputs=["/Constant41_output_0"],  # Output name
        name="/Constant41",   # Node name
        value=constant_tensor                    # Attribute for constant value
    )
    model.graph.node.append(constant_node)


    # for node in graph.node:
    #     if node.name == "/model/layers.0/mlp/Constant_2" or node.name == "/model/layers.0/self_attn/Constant_40":
        # if node.name == "/model/layers.0/mlp/Constant_2" or node.name =="/model/layers.0/mlp/Constant_3" or node.name == "/model/layers.0/mlp/Constant_4" or node.name == "/model/layers.0/self_attn/Constant_40":
            # print(node)  # Print the details of the node
            # print(type(node.attribute.t.raw_data))
            # print(node.attribute['name'])
            # print(node.attribute.type)
            # print(node.attribute.t)

    # onnx.save(model, onnx_model_filename)
    onnx.save_model(model, onnx_model_filename, save_as_external_data=True, all_tensors_to_one_file=True, location="external_weights", size_threshold=1024, convert_attribute=False)
    # onnx.save_model(model, "Phi-3.5-mini-instruct_1l_fixed.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="external_weights", size_threshold=1024, convert_attribute=False)

def run_comparison(self, model_name, use_minimal_model=True, use_tensorrt=True, use_fp16=True, disable_ort_graph_optimization=False):
    start_time = time.time()
    model, pytorch_inputs, ort_inputs = get_model_and_inputs(model_name, use_minimal_model)
    pytorch_output = run_model_in_pytorch(model, pytorch_inputs)
    pytorch_output = pytorch_output.numpy()
    suffix = "_min" if use_minimal_model else ""
    model_file = model_name.split("/")[1] + suffix + ".onnx"
    # Export pytorch model to onnx
    input_names = list(pytorch_inputs.keys())
    # torch.onnx.export(model, pytorch_inputs, model_file)
    # torch.onnx.export(model, (inputs['input_ids'], inputs['attention_mask']), model_file, input_names = ['input_ids', 'attention_mask'], opset_version=17, verbose=True)
    torch.onnx.export(model, pytorch_inputs, model_file, input_names=input_names)
    if model_name == "microsoft/Phi-3.5-mini-instruct":
        fix_phi35_model(model_file)
    providers = get_ep(use_tensorrt, use_fp16)
    ort_output = run_model_in_ort(model_file, ort_inputs, providers, disable_ort_graph_optimization=disable_ort_graph_optimization)
    # ort_output = run_model_in_ort("Phi-3.5-mini-instruct_1l_fixed.onnx", ort_inputs, providers, disable_ort_graph_optimization=disable_ort_graph_optimization)
    print(f"pytorch_output={pytorch_output}")
    print(f"ort_output={ort_output}")
    are_close = np.allclose(pytorch_output, ort_output, rtol=1e-2, atol=1e-2)
    print(f"====\n{model_name}{suffix} FP16={use_fp16} disable_ort_graph_optimization={disable_ort_graph_optimization} pytorch CPU and ORT {providers[0][0]} results are close")
    self.assertTrue(are_close, f"====\n{model_name}{suffix} FP16={use_fp16} disable_ort_graph_optimization={disable_ort_graph_optimization} pytorch CPU and ORT {providers[0][0]} results should be close")
    difference = np.linalg.norm(ort_output - pytorch_output)
    print("Difference:", difference)
    diff = np.abs(ort_output - pytorch_output).mean()
    print(f"Mean absolute difference: {diff}")
    rel_diff = np.abs(ort_output - pytorch_output) / np.abs(pytorch_output + 1e-8)  # Add epsilon to avoid division by zero
    print(f"Max relative difference: {np.max(rel_diff)}")
    end_time = time.time()  # End the timer
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

    def test_resnet_cpu_fp32_wo_opt(self):
        run_comparison(self, "microsoft/resnet-18", use_minimal_model=False, use_tensorrt=False, use_fp16=False, disable_ort_graph_optimization=True)
    
    def test_resnet_cpu_fp32(self):
        run_comparison(self, "microsoft/resnet-18", use_minimal_model=False, use_tensorrt=False, use_fp16=False, disable_ort_graph_optimization=False)

    def test_resnet_cpu_fp32(self):
        run_comparison(self, "microsoft/resnet-18", use_minimal_model=False, use_tensorrt=True, use_fp16=False, disable_ort_graph_optimization=False)

    def test_resnet_trt_fp32(self):
        run_comparison(self, "microsoft/resnet-18", use_minimal_model=False, use_tensorrt=True, use_fp16=True, disable_ort_graph_optimization=False)

    def test_resnet_trt_fp16(self):
        run_comparison(self, "microsoft/resnet-18", use_minimal_model=False, use_tensorrt=True, use_fp16=False, disable_ort_graph_optimization=False)

    def test_resnet50_trt_fp16(self):
        run_comparison(self, "microsoft/resnet-50", use_minimal_model=False, use_tensorrt=True, use_fp16=False, disable_ort_graph_optimization=False)

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
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", use_minimal_model=True, use_tensorrt=False, use_fp16=False, disable_ort_graph_optimization=True)
    
    def test_phi35_1l_cpu_fp32(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", use_minimal_model=True, use_tensorrt=False, use_fp16=False, disable_ort_graph_optimization=False)

    def test_phi35_1l_trt_fp32(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", use_minimal_model=True, use_tensorrt=True, use_fp16=False, disable_ort_graph_optimization=False)

    def test_phi35_1l_trt_fp16(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", use_minimal_model=True, use_tensorrt=True, use_fp16=True, disable_ort_graph_optimization=False)

    def test_phi35_full_trt_fp16(self):
        run_comparison(self, "microsoft/Phi-3.5-mini-instruct", use_minimal_model=False, use_tensorrt=True, use_fp16=True, disable_ort_graph_optimization=False)


if __name__ == "__main__":
    unittest.main()
