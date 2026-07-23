"""Generate tiny base models + a small shared external weights file, then compile
OpenVINO NPU EPContext variants (shared, weightless bin).

Two shape specializations sharing ONE weights.data:
  prefill : input [1, 4, 64]
  iter    : input [1, 1, 64]

Architecture: Relu(MatMul(x, W_i) + b_i) x L  (all OV-native ops).

Outputs into <out>/base and <out>/compiled.
"""
import argparse
import glob
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from windowsml import EpCatalog

EP = "OpenVINOExecutionProvider"
D = 64
L = 4
DATA_NAME = "weights.data"
SHAPES = {"prefill": 4, "iter": 1}
RNG = np.random.default_rng(0)


def write_weights(data_path):
    layout = {}
    scale = 1.0 / np.sqrt(D)
    with open(data_path, "wb") as f:
        for i in range(L):
            W = (RNG.standard_normal((D, D)) * scale).astype(np.float32)
            off = f.tell(); f.write(W.tobytes())
            layout[f"W_{i}"] = (off, W.nbytes, [D, D])
            b = np.zeros((D,), np.float32)
            off = f.tell(); f.write(b.tobytes())
            layout[f"b_{i}"] = (off, b.nbytes, [D])
    return layout


def ext_init(name, shape, offset, length):
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.FLOAT
    t.dims.extend(shape)
    t.data_location = TensorProto.EXTERNAL
    for k, v in (("location", DATA_NAME), ("offset", str(offset)), ("length", str(length))):
        e = t.external_data.add(); e.key, e.value = k, v
    return t


def build_model(seq, layout):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, seq, D])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, seq, D])
    nodes, inits = [], []
    cur = "input"
    for i in range(L):
        woff, wlen, wshape = layout[f"W_{i}"]
        boff, blen, bshape = layout[f"b_{i}"]
        inits += [ext_init(f"W_{i}", wshape, woff, wlen), ext_init(f"b_{i}", bshape, boff, blen)]
        mm, ad = f"mm_{i}", f"add_{i}"
        rl = "output" if i == L - 1 else f"relu_{i}"
        nodes += [helper.make_node("MatMul", [cur, f"W_{i}"], [mm], name=f"MatMul_{i}"),
                  helper.make_node("Add", [mm, f"b_{i}"], [ad], name=f"Add_{i}"),
                  helper.make_node("Relu", [ad], [rl], name=f"Relu_{i}")]
        cur = rl
    g = helper.make_graph(nodes, "tiny_mlp", [inp], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    return m


def register_ep():
    with EpCatalog() as cat:
        ep = next(e for e in cat.find_all_providers() if "openvino" in e.name.lower())
        ep.ensure_ready(); lib = ep.library_path
    ort.register_execution_provider_library(EP, lib)
    return lib


def npu_device():
    devs = [d for d in ort.get_ep_devices() if d.ep_name == EP]
    by = {str(d.device.type).rsplit(".", 1)[-1]: d for d in reversed(devs)}
    return by.get("NPU") or list(by.values())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    base = os.path.join(args.out, "base")
    compiled = os.path.join(args.out, "compiled")
    for d in (base, compiled):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

    # 1) shared weights + base models
    layout = write_weights(os.path.join(base, DATA_NAME))
    for name, seq in SHAPES.items():
        onnx.save(build_model(seq, layout), os.path.join(base, f"{name}.onnx"))
    print(f"base weights.data: {os.path.getsize(os.path.join(base, DATA_NAME))} bytes")

    # 2) compile OV NPU shared-context weightless ctx models
    lib = register_ep()
    print("EP:", lib)
    dev = npu_device()
    # weights must be reachable next to the input model at compile time (they already are in base/)
    names = list(SHAPES)
    live = []
    for i, name in enumerate(names):
        so = ort.SessionOptions(); so.log_severity_level = 3
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_embed_mode", "0")
        so.add_session_config_entry("ep.context_file_path", os.path.join(compiled, f"{name}.ctx.onnx"))
        so.add_session_config_entry("ep.share_ep_contexts", "1")
        if i == len(names) - 1:
            so.add_session_config_entry("ep.stop_share_ep_contexts", "1")
        so.add_provider_for_devices([dev], {})
        live.append(ort.InferenceSession(os.path.join(base, f"{name}.onnx"), sess_options=so))
        print(f"compiled {name}.ctx.onnx")

    print("\ncompiled dir:")
    for f in sorted(glob.glob(os.path.join(compiled, "*"))):
        print(f"  {os.path.getsize(f):>10} B  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
