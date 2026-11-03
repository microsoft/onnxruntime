# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Explicit local-asset qualification; no network or implicit model acquisition."""

import ctypes
import gc
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from onnxruntime_test_python_phi3_demo import demo

import onnxruntime as ort
from onnxruntime import stateful as sf

PROMPT = "Answer briefly: what is 2 + 2?"


def assets_root():
    value = os.environ.get("ORT_STATEFUL_PHI3_ASSETS")
    if not value:
        raise ValueError("Set ORT_STATEFUL_PHI3_ASSETS explicitly; these tests never download models")
    root = Path(value)
    if not root.is_dir():
        raise ValueError(f"Local Phi-3 assets are unavailable: {root}")
    return root


def qualification_package(assets, target, budget):
    source = assets / target
    root = assets / "qualification" / f"{target}-{budget}"
    directory = root / "decoder" / target
    directory.mkdir(parents=True, exist_ok=True)
    for asset in demo.PROFILES[target].assets:
        destination = directory / asset.name
        if not destination.exists():
            destination.hardlink_to(source / "decoder" / target / asset.name)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["additional_metadata"]["phi3_demo"]["max_new_tokens"] = budget
    exits = manifest["chains"]["generate"]["steps"][1]["executor_info"]["ort"]["loop"]["exit_conditions"]
    next(value for value in exits if value["kind"] == "max_generated_tokens")["limit"] = budget
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def programmatic_tokens(package, prompt):
    config = demo.package_configuration(package)
    target = config["target"]
    directory = package / "decoder" / target
    report = demo.inspect_variant(directory, target)
    adapter = demo.LocalTokenizer(directory)
    ids = np.array([adapter.encode(prompt)], dtype=np.int64)
    options = demo.session_options(target, config["device_id"])
    providers = ["CPUExecutionProvider"] if target == "cpu" else None
    first = ort.InferenceSession(
        str(directory / report["filename"]), options, providers=providers, enable_fallback=False
    )
    second = ort.InferenceSession(
        str(directory / report["filename"]), options, providers=providers, enable_fallback=False
    )
    provider = (
        sf.ProviderSelection()
        if target == "cpu"
        else sf.ProviderSelection(
            sf.ProviderMode.REQUIRE, "CUDAExecutionProvider", sf.DeviceType.GPU, config["device_id"]
        )
    )
    element = sf.ElementType.FLOAT32 if target == "cpu" else sf.ElementType.FLOAT16
    bindings = tuple(
        sf.CacheBinding(sf.CacheSpec(value["name"], element, tuple(value["shape"]), 2), value["input"], value["output"])
        for value in report["caches"]
    )
    outputs = tuple(report["outputs"])
    with (
        sf.SessionLoop(
            first,
            sf.LoopConfig(
                outputs,
                (sf.ExitCondition(sf.ExitKind.END_OF_INPUTS),),
                finite_input_names=(report["input_ids"],),
                nonempty_input=report["input_ids"],
                kv_caches=bindings,
                cache_provider=provider,
            ),
        ) as prefill,
        sf.SessionLoop(
            second,
            sf.LoopConfig(
                outputs,
                (
                    sf.ExitCondition(sf.ExitKind.MAX_GENERATED_TOKENS, config["max_new_tokens"]),
                    sf.ExitCondition(sf.ExitKind.EOS, eos_token_ids=tuple(report["eos_token_ids"])),
                ),
                kv_caches=bindings,
                cache_provider=provider,
                sampler_provider=provider,
                generation=sf.GenerationConfig(
                    sf.GenerationPhase.DECODE,
                    report["input_ids"],
                    report["logits"],
                    attention_mask=report["attention_mask"],
                ),
            ),
        ) as decode,
    ):
        chain = sf.SessionChain(
            sf.ChainConfig(
                (
                    sf.ChainComponent(
                        "prefill",
                        prefill,
                        (
                            sf.InputBinding(
                                report["input_ids"], sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "prompt")
                            ),
                            sf.InputBinding(
                                report["attention_mask"], sf.ValueSource(sf.SourceKind.EXTERNAL_INPUT, "mask")
                            ),
                        ),
                    ),
                    sf.ChainComponent(
                        "decode",
                        decode,
                        generation_bootstrap=sf.GenerationBootstrap(
                            sf.ValueSource(sf.SourceKind.COMPONENT_OUTPUT, report["logits"], 0),
                            sf.ValueSource(sf.SourceKind.EXTERNAL_SEQUENCE, "prompt"),
                        ),
                    ),
                ),
                (sf.ChainOutput("tokens", 1, kind=sf.OutputKind.GENERATED_TOKEN_IDS),),
            )
        )
    with chain:
        execution = chain.create_execution({"mask": np.ones(ids.shape, dtype=np.int64)}, sequences={"prompt": [ids]})
    with execution, execution.run() as result:
        return result.get_outputs()["tokens"].numpy().tolist()


class TestPhi3RealAssets(unittest.TestCase):
    def setUp(self):
        if not os.environ.get("ORT_STATEFUL_PHI3_ASSETS"):
            self.skipTest("Explicit real assets are required; no download is performed")
        self.assets = assets_root()

    def test_cpu_real_reference_boundaries_replay_and_construction_parity(self):
        self._qualify("cpu")

    def test_cuda_real_reference_boundaries_replay_and_construction_parity(self):
        self.assertIn("cuda-plugin-ep=1", ort.get_build_info())
        self._qualify("cuda")

    def _qualify(self, target):
        reports = []
        for budget in (0, 1, 16):
            with self.subTest(target=target, budget=budget):
                package = qualification_package(self.assets, target, budget)
                report = demo.validate_generation(package, PROMPT)
                self.assertLessEqual(len(report["token_ids"]), budget)
                self.assertTrue(report["reset_replay_verified"])
                self.assertEqual(report["cache_count"], 64)
                self.assertEqual(report["invocations"], max(1, len(report["token_ids"])))
                if budget == 0:
                    self.assertEqual(report["sampler_provider_calls"] + report["sampler_cpu_calls"], 0)
                elif target == "cuda":
                    self.assertEqual(report["sampler_origin"], "PROVIDER")
                    self.assertEqual(report["sampler_cpu_calls"], 0)
                typed = programmatic_tokens(package, PROMPT)
                self.assertEqual(typed, report["token_ids"])
                reports.append(report)
        destination = os.environ.get("ORT_STATEFUL_PHI3_REPORTS")
        if destination:
            path = Path(destination)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{target}-qualification.json").write_text(json.dumps(reports, indent=2))


def cuda_worker(mode):
    root = qualification_package(assets_root(), "cuda", 3)
    execution, _, prompt_length = demo.create_execution(root, PROMPT, None)
    runtime = None
    if mode == "--cuda-trace-worker":
        with execution.run() as warmup:
            assert warmup.get_info().generated_tokens == 3
        execution.reset()
    initial_calls = execution.get_dispatch_info(
        sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1
    ).provider_calls
    if mode != "--cuda-sanitizer-worker":
        runtime = ctypes.CDLL(os.environ.get("ORT_STATEFUL_CUDART", "libcudart.so.13"))
        for operation in ("cudaProfilerStart", "cudaProfilerStop", "cudaDeviceSynchronize"):
            getattr(runtime, operation).argtypes = []
            getattr(runtime, operation).restype = ctypes.c_int
        if mode == "--cuda-trace-worker":
            assert runtime.cudaProfilerStart() == 0
    with execution:
        result = execution.run()
        info = result.get_info()
        assert info.invocations >= 2, "A bootstrap-only run does not qualify actual decode"
        assert info.generated_tokens == 3
        dispatch = execution.get_dispatch_info(sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1)
        assert dispatch.provider_calls - initial_calls == 3 and dispatch.cpu_calls == 0
        state = execution.get_state().get_outputs(sf.OutputGroup.STATE_VALUES)
        assert len(state) == 64
        for value in state.values():
            metadata = value.get_info()
            assert metadata.device_type == sf.DeviceType.GPU and metadata.device_id == 0
            assert metadata.shape[2] == prompt_length + 2
        retained = state["key_0"]
        if mode == "--cuda-copy-worker":
            assert runtime.cudaProfilerStart() == 0
            cloned = retained.to_ortvalue()
            assert cloned.device_name().lower() == "cuda"
            copied = retained.copy_to_cpu()
            assert copied.numpy().shape[2] == prompt_length + 2
            del cloned, copied
        if runtime is not None:
            assert runtime.cudaDeviceSynchronize() == 0
            assert runtime.cudaProfilerStop() == 0
        tokens = result.get_outputs()["tokens"].numpy()
        assert tokens.shape == (3,)
        result.close()
    copied = retained.copy_to_cpu()
    assert copied.numpy().shape[2] == prompt_length + 2
    retained.close()
    del copied, state
    gc.collect()
    print(
        json.dumps(
            {
                "mode": mode,
                "prompt_tokens": prompt_length,
                "invocations": info.invocations,
                "generated_tokens": info.generated_tokens,
                "cache_bytes": (prompt_length + 2) * 32 * 96 * 2,
            }
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] in (["--cuda-sanitizer-worker"], ["--cuda-trace-worker"], ["--cuda-copy-worker"]):
        cuda_worker(sys.argv[1])
    else:
        unittest.main()
