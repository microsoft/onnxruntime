from onnx import TensorProto, helper, save


def make_stft_model(frame_length_value, frame_step_value, has_window, filename):
    batch = 1
    signal_length = 16
    dft_size = abs(frame_length_value) if frame_length_value != 0 else 4
    onesided_bins = dft_size // 2 + 1
    num_frames = max(1, (signal_length - dft_size) // abs(frame_step_value) + 1) if frame_step_value != 0 else 1

    signal = helper.make_tensor_value_info("signal", TensorProto.FLOAT, [batch, signal_length, 1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, num_frames, onesided_bins, 2])

    frame_step_init = helper.make_tensor("frame_step", TensorProto.INT64, [], [frame_step_value])
    frame_length_init = helper.make_tensor("frame_length", TensorProto.INT64, [], [frame_length_value])

    window_input = "window" if has_window else ""
    initializers = [frame_step_init, frame_length_init]
    if has_window:
        window_init = helper.make_tensor("window", TensorProto.FLOAT, [dft_size], [1.0] * dft_size)
        initializers.append(window_init)

    stft_node = helper.make_node(
        "STFT",
        inputs=["signal", "frame_step", window_input, "frame_length"],
        outputs=["output"],
        onesided=1,
    )

    graph = helper.make_graph(
        [stft_node],
        "stft_test",
        [signal],
        [output],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save(model, filename)


if __name__ == "__main__":
    make_stft_model(-2, 4, False, "stft_negative_frame_length.onnx")
    make_stft_model(4, -2, False, "stft_negative_frame_step.onnx")
    make_stft_model(4, 4, False, "stft_no_window.onnx")
