import argparse
from pathlib import Path
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prepare_hc_projection_silu import DOMAIN, REPOS, fuse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("projection", "up"), default="projection")
    up = parser.parse_args().stage == "up"
    if up:
        from prepare_hc_up_premix import fuse as rewrite
    else:
        rewrite = fuse
    source = REPOS / ("qwen38-int4-hc-projection-silu/model" if up else "qwen38-int4-hc-post-norm/model")
    original = onnx.load(source / "model.onnx", load_external_data=False)
    initializers = {t.name: t for t in original.graph.initializer}
    producers = {o: n for n in original.graph.node for o in n.output}
    directory = REPOS / ("qwen38-int4-hc-up-premix" if up else "qwen38-int4-hc-projection-silu")
    output = directory / "real-weight-fixtures-v2"
    output.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.name != "model.onnx" and path.is_file():
            target = output / path.name
            try:
                target.hardlink_to(path)
            except PermissionError:
                subprocess.run(["sudo", "-n", "ln", str(path), str(target)], check=True)
            assert target.samefile(path)
    random = np.random.default_rng(20261029)
    count = 0
    for split in original.graph.node:
        if split.op_type != ("HyperConnectionPreMixSigmoidF16" if up else "SplitScaledSiLUF16"):
            continue
        matmul = producers[split.input[1 if up else 0]]
        if not up:
            sizes = onnx.TensorProto()
            if split.input[1] in initializers:
                sizes.CopyFrom(initializers[split.input[1]])
            else:
                constant = producers[split.input[1]]
                assert constant.op_type == "Constant"
                sizes.CopyFrom(next(a.t for a in constant.attribute if a.name == "value"))
            sizes_array = numpy_helper.to_array(sizes, base_dir=str(source))
            assert sizes_array.tolist() == [320, 4]
        for stress, magnitude in enumerate((0.125, 1.0, 4.0)):
            weights = [initializers[n] for n in matmul.input[1:]]
            arrays = [numpy_helper.from_array(
                random.normal(0, magnitude, [1, 1, 320 if up else 10240]).astype(np.float16), "x")]
            if up:
                arrays.append(numpy_helper.from_array(
                    random.normal(0, magnitude, [1, 1, 10240]).astype(np.float16), "streams"))
            else:
                arrays.append(numpy_helper.from_array(sizes_array, "sizes"))
            matrix = onnx.NodeProto()
            matrix.CopyFrom(matmul)
            matrix.input[0] = "x"
            matrix.output[0] = "projected"
            activation = onnx.NodeProto()
            activation.CopyFrom(split)
            del activation.input[:]
            activation.input.extend(["streams", "projected"] if up else ["projected", "sizes"])
            del activation.output[:]
            activation.output.extend(["y"] if up else ["activated", "tail"])
            graph = helper.make_graph([matrix, activation], "real_projection", [],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, 2560])] if up else
                [helper.make_tensor_value_info("activated", TensorProto.FLOAT16, [1, 1, 320]),
                 helper.make_tensor_value_info("tail", TensorProto.FLOAT16, [1, 1, 4])],
                weights + arrays)
            reference = helper.make_model(graph, functions=original.functions,
                opset_imports=original.opset_import, ir_version=original.ir_version)
            candidate, fused = rewrite(reference)
            assert fused == 1
            name = f"projection_{count}_stress{stress}"
            for suffix, model in (("", candidate), ("_reference", reference)):
                path = output / (name + suffix + ".onnx")
                onnx.save(model, path)
                onnx.checker.check_model(str(path))
        count += 1
    assert count == (97 if up else 96)
    build = directory / "real-weight-build-v1"
    build.mkdir(exist_ok=False)
    text = Path(__file__).with_name("hc_fusion_compare.cpp").read_text()
    text = text.replace("options.SetIntraOpNumThreads(1);",
        "options.SetIntraOpNumThreads(1);\n"
        "    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);")
    (build / "native_compare.cpp").write_text(text)
    print(f"Prepared {count * 3} real-weight fixture pairs with hardlinked original external data")


if __name__ == "__main__":
    main()
