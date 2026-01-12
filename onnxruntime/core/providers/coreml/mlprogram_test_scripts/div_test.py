import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.utils import save_spec

input_dim = (1,)
output_dim = (1,)


def mlprogram():
    target = ct.target.iOS15

    @mb.program(input_specs=[mb.TensorSpec(shape=input_dim), mb.TensorSpec(shape=input_dim)], opset_version=target)
    def prog(x, y):
        return mb.real_div(x=x, y=y)

    # print(prog)

    # Convert to ML program
    m = ct.convert(prog, minimum_deployment_target=target)

    x = np.array([2], dtype=np.float32)
    y = np.array([2047], dtype=np.float32)

    # spec = m.get_spec()
    # print(spec)

    print(m.predict({"x": x, "y": y}))


# implement Div with coremltools approach of x * (1/y)
def nn():
    input_features = [("x", datatypes.Array(*input_dim)), ("y_inv", datatypes.Array(*input_dim))]
    output_features = [("final", datatypes.Array(*output_dim))]

    # Build a simple neural network with 1 inner product layer
    builder = NeuralNetworkBuilder(input_features, output_features)
    builder.add_elementwise(
        name="x_multiply_inverse_of_y",
        input_names=["x", "y_inv"],
        output_name="final",
        mode="MULTIPLY",
    )

    save_spec(builder.spec, "network.mlmodel")
    m = ct.models.MLModel("network.mlmodel")

    x = np.array([2], dtype=np.float32)
    y = np.array([1 / 2047], dtype=np.float32)
    print(m.predict({"x": x, "y_inv": y}))


def nn_scale():
    input_features = [
        ("x", datatypes.Array(*input_dim)),
        ("y_inv", datatypes.Array(*input_dim)),
        ("z", datatypes.Array(*input_dim)),
    ]
    output_features = [("final", datatypes.Array(*output_dim))]

    builder = NeuralNetworkBuilder(input_features, output_features)

    builder.add_elementwise(
        name="div_implemented_as_x_multiply_inverse_of_y",
        input_names=["x", "y_inv"],
        output_name="div_result",
        mode="MULTIPLY",
    )

    builder.add_elementwise(
        name="apply_scaling_factor",
        input_names=["div_result", "z"],
        output_name="final",
        mode="MULTIPLY",
    )

    from coremltools.models.utils import save_spec  # noqa: PLC0415

    save_spec(builder.spec, "network.mlmodel")
    m = ct.models.MLModel("network.mlmodel")

    a = 2
    b = 2047
    # scaling factor to test working around coremltools inaccuracy.
    # weirdly even a scaling factor of 1 fixes the problem from https://github.com/microsoft/onnxruntime/issues/21170
    c = 1000

    x = np.array([a], dtype=np.float32)
    y = np.array([1 / b / c], dtype=np.float32)
    z = np.array([c], dtype=np.float32)
    print(m.predict({"x": x, "y_inv": y, "z": z}))


print("NN")
nn()

print("\nNN with scaling")
nn_scale()

print("\nML Program")
mlprogram()
