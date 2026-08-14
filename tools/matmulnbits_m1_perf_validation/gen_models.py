#!/usr/bin/env python3
"""Generate MatMulNBits ONNX models (M=1) for A10 perf-validation via onnxruntime_perf_test.

Each model is a single-node graph: Y = MatMulNBits(A, B_quant, scales, zero_points)
with fp32 dtype, K/N/block_size matching one of the representative shapes used in
microsoft/onnxruntime PR #31988. Weights are quantized with the *real* RTN weight-only
quantizer (onnxruntime.quantization.matmul_nbits_quantizer) so the packed-B / scales /
zero_points layout is guaranteed correct (bit-for-bit what production quantization
produces), removing any risk of a hand-rolled packing bug.
"""
import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig
import os

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "models"))
args = parser.parse_args()
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# (label, N, K, block_size)
SHAPES = [
    ("narrow_n256_k1024_bs128",   256,  1024, 128),
    ("narrow_n384_k1024_bs128",   384,  1024, 128),
    ("boundary_n576_k1024_bs128", 576,  1024, 128),
    ("wide_n4096_k4096_bs128",   4096,  4096, 128),
    ("wide_n8192_k4096_bs128",   8192,  4096, 128),
    ("wide_n11008_k4096_bs128", 11008,  4096, 128),
    ("wide_n14336_k4096_bs128", 14336,  4096, 128),
    ("narrow_n256_k4096_bs128",   256,  4096, 128),
    ("narrow_n256_k11008_bs128",  256, 11008, 128),
    ("wide_n8192_k11008_bs128",  8192, 11008, 128),
    ("narrow_n256_k1024_bs32",    256,  1024, 32),
    ("narrow_n384_k1024_bs64",    384,  1024, 64),
]

RNG = np.random.default_rng(1234)


def make_float_matmul_model(n, k, path):
    """A@W^T style: single MatMul(A[1,K], W[K,N]) -> Y[1,N], to be quantized."""
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, k])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n])
    w = RNG.standard_normal((k, n)).astype(np.float32) * 0.05
    w_init = numpy_helper.from_array(w, name="W")
    node = helper.make_node("MatMul", ["A", "W"], ["Y"], name="mm")
    graph = helper.make_graph([node], "g", [a], [y], initializer=[w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.save(model, path)


def quantize(src_path, dst_path, block_size):
    quant_config = RTNWeightOnlyQuantConfig()
    quantizer = MatMulNBitsQuantizer(
        model=src_path,
        block_size=block_size,
        is_symmetric=False,
        accuracy_level=0,
        algo_config=quant_config,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(dst_path, use_external_data_format=False)


def verify_and_report(path, n, k):
    m = onnx.load(path)
    ops = sorted({node.op_type for node in m.graph.node})
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    a = RNG.standard_normal((1, k)).astype(np.float32) * 0.1
    (y,) = sess.run(None, {"A": a})
    assert y.shape == (1, n), f"bad output shape {y.shape}"
    assert np.isfinite(y).all(), "non-finite output"
    return ops, float(np.abs(y).mean())


def main():
    manifest = []
    for label, n, k, bs in SHAPES:
        float_path = os.path.join(OUT_DIR, f"_tmp_float_{label}.onnx")
        quant_path = os.path.join(OUT_DIR, f"matmulnbits_m1_{label}.onnx")
        make_float_matmul_model(n, k, float_path)
        quantize(float_path, quant_path, bs)
        ops, ymean = verify_and_report(quant_path, n, k)
        os.remove(float_path)
        assert "MatMulNBits" in ops, f"{label}: quantizer did not produce MatMulNBits, got {ops}"
        size = os.path.getsize(quant_path)
        manifest.append((label, n, k, bs, size, ymean))
        print(f"OK  {label:28s} N={n:6d} K={k:6d} bs={bs:4d}  size={size/1024:8.1f}KiB  |y|mean={ymean:.4f}")

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w") as f:
        f.write("label,n,k,block_size,file_bytes,y_abs_mean\n")
        for row in manifest:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nGenerated {len(manifest)} models in {OUT_DIR}")


if __name__ == "__main__":
    main()
