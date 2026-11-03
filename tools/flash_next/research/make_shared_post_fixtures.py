import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from fuse_shared_post import fuse
from test_shared_post_fusion import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative-only", action="store_true")
    parser.add_argument("--extended-only", action="store_true")
    parser.add_argument("--layouts-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent / "shared-post-fixtures"
    root.mkdir(exist_ok=True)
    if args.negative_only:
        for name, tensor_name, shape in [
            ("gate_last", "SharedGate", [1, 2, 2]),
            ("shared_width", "Shared", [1, 2, 8]),
        ]:
            model = onnx.load(root / "shared_m2_h7_c4_stress0.onnx")
            tensor = next(v for v in model.graph.initializer if v.name == tensor_name)
            tensor.CopyFrom(numpy_helper.from_array(np.zeros(shape, np.float16), tensor_name))
            target = root / ("negative_" + name + ".onnx")
            reference = root / ("negative_" + name + "_reference.onnx")
            if target.exists() or reference.exists():
                raise FileExistsError(name)
            onnx.save_model(model, target)
            reference.hardlink_to(root / "shared_m2_h7_c4_stress0_reference.onnx")
        print("Created two unsupported shared-post shape fixtures")
        return
    rng = np.random.default_rng(20261027)
    layouts = [(2, 8, 4, "feature_grouped"), (1, 2560, 4, "branch_singleton")]
    cases = (layouts if args.layouts_only else
             [(3, 12, 4, "branch"), *layouts]
             if args.extended_only else
             [(1, 1, 1, "branch"), (2, 7, 4, "branch"), (1, 2560, 4, "branch"),
              (16, 2560, 4, "branch"), (0, 7, 4, "branch")])
    for rows, hidden, branches, gate_kind in cases:
        for stress in (False, True):
            original = fixture(rows, hidden, branches)
            if gate_kind != "branch":
                gate = next(v for v in original.graph.input if v.name == "G")
                shape = [1, rows, branches, hidden] if gate_kind == "feature_grouped" else [1, rows, branches, 1]
                gate.CopyFrom(helper.make_tensor_value_info("G", TensorProto.FLOAT16, shape))
            for value in original.graph.input:
                shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
                data = rng.uniform(-5, 5, shape).astype(np.float16)
                if stress:
                    data = rng.choice(np.array([-16, -8, -1.001, -0.000061, 0, 0.000061,
                                                1.001, 8, 16], np.float16), shape)
                original.graph.initializer.append(numpy_helper.from_array(data, value.name))
            del original.graph.input[:]
            model, count = fuse(original)
            assert count == 1
            name = f"shared_m{rows}_h{hidden}_c{branches}_stress{int(stress)}"
            if args.extended_only or args.layouts_only:
                name += "_" + gate_kind
            for graph, suffix in [(model, ""), (original, "_reference")]:
                onnx.checker.check_model(graph)
                path = root / (name + suffix + ".onnx")
                if path.exists():
                    raise FileExistsError(path)
                onnx.save_model(graph, path)
    print(f"Created {len(cases) * 2} shared-post fused/reference fixture pairs")


if __name__ == "__main__":
    main()
