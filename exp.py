import os
import sys

sys.path.insert(0, "/home/guangyunhan/onnxruntime/build_rocm/Release/build/lib")


import numpy as np
import onnx

import onnxruntime as ort

input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT16, ["batchsize", 512, 768])
attn_mask = onnx.helper.make_tensor_value_info("attn_mask", onnx.TensorProto.INT32, ["batchsize", 512])
output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, ["batchsize", 512, 768])

np.random.seed(1)
qkv_weight = onnx.helper.make_tensor("qkv_weight", onnx.TensorProto.FLOAT16, [768, 2304], np.random.randn(*[768, 2304]))
# qkv_weight = onnx.helper.make_tensor("qkv_weight", onnx.TensorProto.FLOAT16, [768, 2304], np.zeros([768, 2304]))
qkv_bias = onnx.helper.make_tensor("qkv_bias", onnx.TensorProto.FLOAT16, [2304], np.random.random([2304]))
# qkv_bias = onnx.helper.make_tensor("qkv_bias", onnx.TensorProto.FLOAT16, [2304], np.zeros([2304]))


node = onnx.helper.make_node(
    "Attention",
    inputs=["input", "qkv_weight", "qkv_bias", "attn_mask"],
    outputs=["output"],
    domain="com.microsoft",
    num_heads=12,
)

graph = onnx.helper.make_graph([node], "Attn", [input, attn_mask], [output], initializer=[qkv_weight, qkv_bias])

model = onnx.helper.make_model(
    graph,
    producer_name="tmp",
    opset_imports=[
        onnx.helper.make_opsetid("com.microsoft", 1),
        onnx.helper.make_opsetid("ai.onnx.ml", 1),
        onnx.helper.make_opsetid("", 14),
    ],
)

sess = ort.InferenceSession(
    model.SerializeToString(), providers=[("ROCMExecutionProvider", {"tunable_op_enabled": "0"})]
)


input = np.random.randn(64, 512, 768)
input = input.astype(np.float16)

os.environ["ORT_ATTENTION_USE_GEMM_RCR_BIAS_PERMUTE"] = sys.argv[1]
os.environ["ORT_ATTENTION_USE_BATCHED_GEMM_SOFTMAX_GEMM_PERMUTE"] = sys.argv[2]

for i in range(10):
    print(i)
    _ = sess.run(
        output_names=[node.name for node in sess.get_outputs()],
        input_feed={"input": input, "attn_mask": np.ones([64, 512], dtype=np.int32)},
    )[0]
