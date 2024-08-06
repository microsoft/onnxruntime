import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

target = ct.target.iOS15

a_shape = (1, 1, 3, 3)


@mb.program(
    input_specs=[mb.TensorSpec(shape=a_shape), mb.TensorSpec(shape=a_shape), mb.TensorSpec(shape=a_shape)],
    opset_version=target,
)
def prog(x, y, z):
    axis = mb.const(val=1)
    interleave = mb.const(val=False)
    z = mb.concat(values=(x, y, z), axis=axis, interleave=interleave)
    return z


print(prog)

# Convert to ML program
m = ct.convert(prog, minimum_deployment_target=target, compute_precision=ct.precision.FLOAT32)

x = np.random.rand(*a_shape)
y = np.random.rand(*a_shape)
z = np.random.rand(*a_shape)

# spec = m.get_spec()
# print(spec)

print(m.predict({"x": x, "y": y, "z": z}))
