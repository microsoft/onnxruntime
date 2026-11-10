# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Tiny native feedback, decoder-bootstrap and DDIM examples; no model downloads."""

import argparse

import numpy as np
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort
from onnxruntime import stateful as sf


def session(graph, provider):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if provider == "CUDAExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(model.SerializeToString(), options, providers=[provider], enable_fallback=False)


def feedback(provider):
    model = session(
        helper.make_graph(
            [helper.make_node("Add", ["x", "past"], ["present"])],
            "feedback",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("past", TensorProto.FLOAT, []),
            ],
            [helper.make_tensor_value_info("present", TensorProto.FLOAT, [])],
        ),
        provider,
    )
    config = sf.LoopConfig(
        ("present",),
        (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
        finite_input_names=("x",),
        feedback=(sf.FeedbackBinding("past", "present"),),
    )
    batches = [{"x": np.array(value, dtype=np.float32)} for value in (2, 3)]
    with sf.SessionLoop(model, config) as loop:
        execution = loop.create_execution({"past": np.array(0, dtype=np.float32)}, batches=batches)
    with execution, execution.run() as result:
        actual = result.get_outputs()["present"].copy_to_cpu().numpy().item()
        assert actual == 5
        print("Feedback:", actual)


def bootstrap(provider):
    scores = np.array([[0, 4, 1, -2], [0, 1, 5, -2], [-1, 1, 2, 6], [6, 2, 1, 0]], dtype=np.float32)
    model = session(
        helper.make_graph(
            [helper.make_node("Gather", ["scores", "token"], ["logits"], axis=0)],
            "decoder",
            [helper.make_tensor_value_info("token", TensorProto.INT64, [1, "S"])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "S", 4])],
            [numpy_helper.from_array(scores, "scores")],
        ),
        provider,
    )
    with (
        sf.SessionLoop(
            model,
            sf.LoopConfig(
                ("logits",),
                (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
                finite_input_names=("token",),
                nonempty_input="token",
            ),
        ) as prefill,
        sf.SessionLoop(
            model,
            sf.LoopConfig(
                ("logits",),
                (sf.ExitCondition(sf.ExitKind.MAX_GENERATED_TOKENS, 3),),
                generation=sf.GenerationConfig(sf.GenerationPhase.DECODE, "token", "logits"),
            ),
        ) as decode,
    ):
        chain = sf.SessionChain(
            sf.ChainConfig(
                (
                    sf.ChainComponent(
                        "prefill",
                        prefill,
                        (sf.InputBinding("token", sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "prompt")),),
                    ),
                    sf.ChainComponent(
                        "decode",
                        decode,
                        generation_bootstrap=sf.GenerationBootstrap(
                            sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, "logits", 0),
                            sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "prompt"),
                        ),
                    ),
                ),
                (sf.ChainOutput("tokens", 1, kind=sf.OutputKind.GENERATED_TOKEN_IDS),),
            )
        )
    with chain:
        execution = chain.create_execution(sequences={"prompt": [np.array([[0, 1]], dtype=np.int64)]})
    with execution:
        prefilled = execution.step()
        assert prefilled.get_info().executed
        sampled = execution.step()
        assert not sampled.get_info().executed
        assert sampled.get_info().generated_token == 2
        with execution.run() as result:
            actual = result.get_outputs()["tokens"].numpy()
            np.testing.assert_array_equal(actual, [2, 3, 0])
            assert result.get_info().invocations == 3
            print("Sampler-free prefill / bootstrap decode:", actual.tolist())


def ddim(provider, mode):
    config = sf.DdimConfig(
        latent_shape=(1, 1, 1, 3),
        num_inference_steps=2,
        guidance_scale=7.5,
        latent_input="sample",
        timestep_input="time",
        conditioning_input="conditioning",
        prediction_output="epsilon",
        initialization_input="init",
        positive_conditioning_input="positive",
        negative_conditioning_input="negative",
        latent_output="latents",
        guidance_mode=mode,
    )
    batch = 2 if mode == sf.GuidanceMode.BATCHED else 1
    shape = [batch, 1, 1, 3]
    model = session(
        helper.make_graph(
            [
                helper.make_node("Mul", ["sample", "latent_scale"], ["scaled_latent"]),
                helper.make_node("Mul", ["time", "time_scale"], ["scaled_time"]),
                helper.make_node("Add", ["scaled_latent", "scaled_time"], ["time_prediction"]),
                helper.make_node("Add", ["time_prediction", "conditioning"], ["epsilon"]),
            ],
            "denoiser",
            [
                helper.make_tensor_value_info("sample", TensorProto.FLOAT, shape),
                helper.make_tensor_value_info("time", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("conditioning", TensorProto.FLOAT, [batch, 1, 1, 1]),
            ],
            [helper.make_tensor_value_info("epsilon", TensorProto.FLOAT, shape)],
            [
                numpy_helper.from_array(np.array(0.125, dtype=np.float32), "latent_scale"),
                numpy_helper.from_array(np.array(0.0001, dtype=np.float32), "time_scale"),
            ],
        ),
        provider,
    )
    with sf.SessionLoop(
        model, sf.LoopConfig(("latents",), (sf.ExitCondition(sf.ExitKind.MAX_ITERATIONS, 2),), ddim=config)
    ) as loop:
        execution = loop.create_execution(
            {
                "init": np.array([0.125, -0.25, 1], dtype=np.float32).reshape(config.latent_shape),
                "positive": np.full((1, 1, 1, 1), 0.25, dtype=np.float32),
                "negative": np.full((1, 1, 1, 1), -0.5, dtype=np.float32),
            }
        )
    with execution, execution.run() as result:
        actual = result.get_outputs()["latents"].numpy().ravel()
        np.testing.assert_allclose(actual, [-8.022068, -8.661748, -6.5294814], rtol=1e-4, atol=1e-4)
        assert result.get_info().invocations == (2 if mode == sf.GuidanceMode.BATCHED else 4)
        print(f"DDIM {mode.name}:", actual.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"), default="CPUExecutionProvider"
    )
    provider = parser.parse_args().provider
    feedback(provider)
    bootstrap(provider)
    for mode in sf.GuidanceMode:
        ddim(provider, mode)


if __name__ == "__main__":
    main()
