"""Generate STFT test models with invalid initializer values for security regression tests."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_stft_model(frame_length_value, frame_step_value, filename):
    """Create a minimal STFT model with given frame_length and frame_step constants."""
    batch = 1
    signal_length = 16
    dft_size = abs(frame_length_value) if frame_length_value != 0 else 4
    onesided_bins = dft_size // 2 + 1
    num_frames = max(1, (signal_length - dft_size) // abs(frame_step_value) + 1) if frame_step_value != 0 else 1

    # Inputs
    signal = helper.make_tensor_value_info("signal", TensorProto.FLOAT, [batch, signal_length, 1])

    # Output (shape must have concrete dims for transformer to fire)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, num_frames, onesided_bins, 2])

    # Constant initializers
    frame_step_init = numpy_helper.from_array(np.array(frame_step_value, dtype=np.int64), name="frame_step")
    frame_length_init = numpy_helper.from_array(np.array(frame_length_value, dtype=np.int64), name="frame_length")

    # STFT node
    stft_node = helper.make_node(
        "STFT",
        inputs=["signal", "frame_step", "", "frame_length"],
        outputs=["output"],
        onesided=1,
    )

    graph = helper.make_graph(
        [stft_node],
        "stft_test",
        [signal],
        [output],
        initializer=[frame_step_init, frame_length_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, filename)


if __name__ == "__main__":
    make_stft_model(frame_length_value=-2, frame_step_value=4, filename="stft_negative_frame_length.onnx")
    make_stft_model(frame_length_value=4, frame_step_value=-2, filename="stft_negative_frame_step.onnx")
    print("Generated test models.")
