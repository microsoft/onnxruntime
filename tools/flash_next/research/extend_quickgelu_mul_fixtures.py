from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from fuse_quickgelu_mul import DOMAIN, OP, definition

root = Path(__file__).parent / "quickgelu-mul-resumed-fixtures"
root.mkdir(exist_ok=True)
rng = np.random.default_rng(20261027)
for shape in ([], [2, 7], [16, 7], [2, 2560], [16, 2560]):
    for stress in (False, True):
        x = rng.uniform(-5, 5, shape).astype(np.float16)
        up = rng.uniform(-2, 2, shape).astype(np.float16)
        if stress:
            x = rng.choice(np.array([-65504, -100, -16, -8, -1.001, -0.000061,
                                     -0., 0., 0.000061, 1.001, 8, 16, 100, 65504], np.float16), shape)
            up = rng.choice(np.array([-0.5, 0, 0.5], np.float16), shape)
        name = "qgm_resumed_" + ("scalar" if not shape else "_".join(map(str, shape))) + f"_stress{int(stress)}"
        initializers = [numpy_helper.from_array(a, n) for a, n in [(x, "X"), (up, "Up")]]
        output = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, shape)
        refs = [helper.make_node("QuickGelu", ["X"], ["Activated"], domain=DOMAIN, alpha=1.0),
                helper.make_node("Mul", ["Activated", "Up"], ["Y"])]
        for suffix, nodes in [("", [helper.make_node(OP, ["X", "Up"], ["Y"], domain=DOMAIN)]),
                              ("_reference", refs)]:
            model = helper.make_model(helper.make_graph(nodes, name, [], [output], initializer=initializers),
                                      opset_imports=[helper.make_opsetid("", 21),
                                                     helper.make_opsetid(DOMAIN, 1)],
                                      functions=[] if suffix else [definition(21)], ir_version=10)
            path = root / (name + suffix + ".onnx")
            if path.exists():
                raise FileExistsError(path)
            onnx.checker.check_model(model)
            onnx.save_model(model, path)
for name, up_shape in [("broadcast", [2, 1]), ("rank", [7])]:
    shape = [2, 7]
    original = onnx.load(root / "qgm_resumed_2_7_stress0.onnx")
    up = next(v for v in original.graph.initializer if v.name == "Up")
    up.CopyFrom(numpy_helper.from_array(np.ones(up_shape, np.float16), "Up"))
    path = root / f"negative_{name}.onnx"
    if path.exists():
        raise FileExistsError(path)
    onnx.save_model(original, path)
    onnx.save_model(original, root / f"negative_{name}_reference.onnx")
print("PASS generated 10 extended fixed-alpha1 pairs and 2 shape guards")
