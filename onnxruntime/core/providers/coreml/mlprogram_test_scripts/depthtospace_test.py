import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

target = ct.target.iOS15

# replicate example from https://github.com/onnx/onnx/blob/main/docs/Operators.md#depthtospace
# to prove CoreML mode is DCR
x_shape = (1, 8, 2, 3)


@mb.program(input_specs=[mb.TensorSpec(shape=x_shape)], opset_version=target)
def prog(x):
    block_size = mb.const(name="block_size", val=2)
    z = mb.depth_to_space(x=x, block_size=block_size)
    return z


print(prog)

# Convert to ML program
m = ct.convert(prog, minimum_deployment_target=target, compute_precision=ct.precision.FLOAT32)

# spec = m.get_spec()
# print(spec)

m.save("DepthToSpace.mlpackage")

# also check for differences between CPU_ONLY and ALL
m_cpu = ct.models.MLModel("DepthToSpace.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
m_all = ct.models.MLModel("DepthToSpace.mlpackage", compute_units=ct.ComputeUnit.ALL)

x = np.array(
    [
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
            [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
            [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
            [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
            [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
            [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
            [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
        ]
    ]
).astype(np.float32)

print("CPU_ONLY")
print(m_cpu.predict({"x": x}))
print("ALL")
print(m_all.predict({"x": x}))
