import argparse
from pathlib import Path

import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/fixtures-v8-fixed-shapes"))
    root = parser.parse_args().output_dir
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    for rows in (1, 2, 16):
        cases.extend([
            (f"quick_rows{rows}", "QuickGeluMulF16", [rows, 1, 1024],
             [("x", [rows, 1, 1024]), ("up", [rows, 1, 1024])], {}, ["y"]),
            (f"gated_rows{rows}", "GatedRMSNorm", [rows, 1, 5120],
             [("x", [rows, 1, 5120]), ("scale", [128]), ("gate", [rows, 1, 5120])],
             {"activation": "silu", "epsilon": 1e-5}, ["y"]),
            (f"split_rows{rows}", "SplitScaledSiLUF16", [rows, 1920],
             [("x", [rows, 1920]), ("sizes", [2])], {"alpha": 0.25}, ["activated", "tail"]),
            (f"premix_rows{rows}", "HyperConnectionPreMixSigmoidF16", [rows, 1, 2560],
             [("streams", [rows, 1, 10240]), ("pre_mix", [rows, 1, 4])],
             {"num_branches": 4, "reduction_scale": 0.25}, ["y"]),
            (f"postnorm_rows{rows}", "HCPostNormF16", [rows, 1, 10240],
             [("streams", [rows, 1, 10240]), ("branch", [rows, 1, 2560]),
              ("gate", [rows, 1, 4]), ("scale", [4, 2560])],
             {"epsilon": 1e-6}, ["y", "normalized"]),
            (f"shared_postnorm_rows{rows}", "HCSharedPostNormF16", [rows, 1, 10240],
             [("streams", [rows, 1, 10240]), ("branch", [rows, 1, 2560]),
              ("gate", [rows, 1, 4]), ("scale", [2560]),
              ("shared_output", [rows, 1, 2560]), ("shared_gate", [rows, 1, 1])],
             {"epsilon": 1e-5}, ["y", "normalized"]),
        ])
    cases.extend([
        ("quick_tail", "QuickGeluMulF16", [1, 257], [("x", [1, 257]), ("up", [1, 257])], {}, ["y"]),
        ("gated_sigmoid", "GatedRMSNorm", [1, 65],
         [("x", [1, 65]), ("scale", [65]), ("gate", [1, 65])],
         {"activation": "sigmoid", "epsilon": 1e-3}, ["y"]),
        ("premix_mixed_f32", "HyperConnectionPreMix", [1, 1, 2560],
         [("streams", [1, 1, 4, 2560]), ("pre_mix", [1, 1, 4])],
         {"num_branches": 0, "reduction_scale": 0.5}, ["y"]),
    ])
    for name, op, shape, inputs, attrs, outputs in cases:
        graph_inputs = [helper.make_tensor_value_info(n, TensorProto.INT64 if n == "sizes"
                                                      else TensorProto.FLOAT if name == "premix_mixed_f32" and n == "pre_mix"
                                                      else TensorProto.FLOAT16, s) for n, s in inputs]
        graph_outputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT16,
                         [shape[0], None] if op == "SplitScaledSiLUF16" else shape) for n in outputs]
        model = helper.make_model(helper.make_graph(
            [helper.make_node(op, [n for n, _ in inputs], outputs, domain="com.microsoft", **attrs)],
            name, graph_inputs, graph_outputs),
            opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
            ir_version=10)
        if op in ("QuickGeluMulF16", "SplitScaledSiLUF16", "HyperConnectionPreMixSigmoidF16",
                  "HCPostNormF16", "HCSharedPostNormF16"):
            evidence = Path(__file__).parent
            reference = (evidence / "quickgelu-mul-fixtures/qgm_1_stress0.onnx"
                         if op == "QuickGeluMulF16" else
                         evidence / "split-silu-fixtures/split_silu_m1_p320_s4_a0.25.onnx"
                         if op == "SplitScaledSiLUF16" else
                         Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm/model/model.onnx"))
            model.functions.extend(f for f in onnx.load(reference, load_external_data=False).functions if f.name == op)
            assert len(model.functions) == 1
        onnx.checker.check_model(model)
        onnx.save(model, root / (name + ".onnx"))
    generator = np.random.default_rng(20261029)
    for k, n in ((2560, 512), (2560, 640), (10240, 324)):
        weights = generator.integers(0, 256, size=(n, k // 32, 16), dtype=np.uint8)
        scales = generator.uniform(0.001, 0.01, size=(n, k // 32)).astype(np.float16)
        for rows in (1, 2, 16):
            name = f"matmul_k{k}_n{n}_rows{rows}"
            model = helper.make_model(helper.make_graph(
                [helper.make_node("MatMulNBits", ["x", "weights", "scales"], ["y"],
                                  domain="com.microsoft", K=k, N=n, bits=4, block_size=32)],
                name, [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [rows, 1, k])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, 1, n])],
                [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(scales, "scales")]),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10)
            onnx.checker.check_model(model)
            onnx.save(model, root / (name + ".onnx"))
    print(f"Prepared {len(cases) + 9} dynamic-input fixtures in {root}")


if __name__ == "__main__":
    main()
