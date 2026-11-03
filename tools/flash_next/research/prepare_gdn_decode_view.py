import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    root = Path("/home/sayanshaw/repos/qwen38-int4-gdn-decode-view")
    output = root / "fixtures"
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261029)
    manifest = []
    for batch, seq in ((1, 1), (1, 2), (2, 1)):
        for heads, value_heads, dk, dv in ((1, 2, 7, 5), (2, 4, 33, 17), (16, 48, 128, 128)):
            for norm in (0, 1):
                for dtype in (np.float16, np.float32):
                    name = f"gdn_b{batch}_s{seq}_h{heads}_dk{dk}_dv{dv}_n{norm}_{np.dtype(dtype).name}"
                    shape = (batch, seq, 2 * heads * dk + value_heads * dv)
                    initializers = [
                        numpy_helper.from_array(rng.uniform(-0.2, 0.2, shape).astype(dtype), "qkv"),
                        numpy_helper.from_array(rng.uniform(-0.4, 0.4, (batch, seq, value_heads)).astype(np.float32), "decay"),
                        numpy_helper.from_array(rng.uniform(-0.4, 0.4, (batch, seq, value_heads)).astype(np.float32), "beta"),
                        numpy_helper.from_array(rng.uniform(-0.1, 0.1, (batch, value_heads, dv, dk)).astype(np.float32), "state"),
                        numpy_helper.from_array(rng.uniform(-1, 1, value_heads).astype(np.float32), "a_log"),
                        numpy_helper.from_array(rng.uniform(-0.4, 0.4, value_heads).astype(np.float32), "bias"),
                    ]
                    node = helper.make_node(
                        "GatedDeltaNet", ["qkv", "", "", "", "decay", "beta", "state", "a_log", "bias"],
                        ["y", "present"], domain="com.microsoft", update_rule="gated_delta",
                        gate_activation="qwen", beta_activation="sigmoid", qk_l2_norm=norm)
                    outputs = [
                        helper.make_tensor_value_info("y", TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT,
                                                      [batch, seq, value_heads, dv]),
                        helper.make_tensor_value_info("present", TensorProto.FLOAT, [batch, value_heads, dv, dk])]
                    model = helper.make_model(helper.make_graph([node], name, [], outputs, initializer=initializers),
                                              opset_imports=[helper.make_opsetid("", 21),
                                                             helper.make_opsetid("com.microsoft", 1)], ir_version=10)
                    for suffix in ("", "_reference"):
                        onnx.save(model, output / (name + suffix + ".onnx"))
                    manifest.append(dict(fixture=name + ".onnx", decode_view=batch * seq == 1,
                                         elements_per_run=batch * seq * value_heads * dv +
                                         batch * value_heads * dv * dk))
    (root / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(fixtures=len(manifest), view_cases=sum(f["decode_view"] for f in manifest))))


if __name__ == "__main__":
    main()
