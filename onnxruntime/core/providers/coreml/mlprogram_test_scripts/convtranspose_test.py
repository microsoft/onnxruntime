import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

target = ct.target.iOS15

x_shape = (1, 3, 4, 4)
w_shape = (3, 3, 3, 3)


@mb.program(input_specs=[mb.TensorSpec(shape=x_shape)], opset_version=target)
def prog(x):
    weight = mb.const(name="weight", val=np.ones(w_shape, dtype=np.float32))
    output_shape = mb.const(name="output_shape", val=np.array([1, 3, 4, 4]))
    # pad = mb.const(val=np.zeros((4), dtype=np.int32))
    strides = mb.const(name="strides", val=np.ones((2), dtype=np.int32))
    dilations = mb.const(name="dilations", val=np.ones((2), dtype=np.int32))
    z = mb.conv_transpose(
        x=x, weight=weight, strides=strides, dilations=dilations, output_shape=output_shape
    )  # , pad=pad

    return z


print(prog)

# Convert to ML program
m = ct.convert(prog, minimum_deployment_target=target, compute_precision=ct.precision.FLOAT32)

# spec = m.get_spec()
# print(spec)

m.save("ConvTranspose.mlpackage")
# construct MLModel with compute_units=ComputeUnit.CPU and run predict
m_cpu = ct.models.MLModel("ConvTranspose.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
m_all = ct.models.MLModel("ConvTranspose.mlpackage", compute_units=ct.ComputeUnit.ALL)

x = np.ones(x_shape, dtype=np.float32)
print("CPU_ONLY")
print(m_cpu.predict({"x": x}))
print("ALL")
print(m_all.predict({"x": x}))
