# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Offline Phi-3 demo tests; real weights are never downloaded by this suite."""

import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_PATH = Path(
    os.environ.get("ORT_STATEFUL_PHI3_DEMO", Path(__file__).resolve().parents[3] / "samples/python/phi3_demo.py")
)
_SPEC = importlib.util.spec_from_file_location("phi3_demo", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = demo
with mock.patch.object(sys, "path", [str(_PATH.parent), *sys.path]):
    _SPEC.loader.exec_module(demo)


class FakeAdapter:
    def encode(self, text, *, raw=False):
        return [2] if text == "eos" else [0, 1]

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens if token != 3)


@contextmanager
def toy_assets():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "cpu/decoder/cpu"
        directory.mkdir(parents=True)
        original = demo.PROFILES["cpu"]
        nodes = [
            helper.make_node("Constant", [], ["axis"], value=numpy_helper.from_array(np.array([1], dtype=np.int64))),
            helper.make_node(
                "Constant", [], ["scale"], value=numpy_helper.from_array(np.array(0.001, dtype=np.float32))
            ),
            helper.make_node(
                "Constant",
                [],
                ["cache_shape"],
                value=numpy_helper.from_array(np.array([1, 1, -1, 1], dtype=np.int64)),
            ),
            helper.make_node("Gather", ["scores", "input_ids"], ["raw_logits"], axis=0),
            helper.make_node("ReduceSum", ["attention_mask", "axis"], ["mask_sum"], keepdims=1),
            helper.make_node("Cast", ["mask_sum"], ["float_mask"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["float_mask", "scale"], ["mask_bias"]),
            helper.make_node("Add", ["raw_logits", "mask_bias"], ["logits"]),
            helper.make_node("Cast", ["input_ids"], ["float_tokens"], to=TensorProto.FLOAT),
            helper.make_node("Reshape", ["float_tokens", "cache_shape"], ["new_tokens"]),
        ]
        inputs = [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "S"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "T"]),
        ]
        outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "S", 4])]
        for layer in range(2):
            for kind in ("key", "value"):
                name = f"past_key_values.{layer}.{kind}"
                present = f"present.{layer}.{kind}"
                inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, "P", 1]))
                outputs.append(helper.make_tensor_value_info(present, TensorProto.FLOAT, [1, 1, "T", 1]))
                nodes.append(helper.make_node("Concat", [name, "new_tokens"], [present], axis=2))
        graph = helper.make_graph(
            nodes,
            "tiny_phi3",
            inputs,
            outputs,
            [
                numpy_helper.from_array(
                    np.array([[0, 4, 1, -2], [0, 1, 5, -2], [-1, 1, 2, 6], [6, 2, 1, 0]], dtype=np.float32), "scores"
                ),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
        onnx.save_model(
            model,
            directory / original.filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=original.filename + ".data",
            size_threshold=0,
        )
        config = {
            "model": {
                "context_length": 32,
                "vocab_size": 4,
                "eos_token_id": [3],
                "decoder": {
                    "filename": original.filename,
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "head_size": 1,
                    "inputs": {
                        "input_ids": "input_ids",
                        "attention_mask": "attention_mask",
                        "past_key_names": "past_key_values.%d.key",
                        "past_value_names": "past_key_values.%d.value",
                    },
                    "outputs": {
                        "logits": "logits",
                        "present_key_names": "present.%d.key",
                        "present_value_names": "present.%d.value",
                    },
                },
            }
        }
        (directory / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")
        (directory / "LICENSE").write_bytes(b"fixture license")
        files = []
        for path in sorted(directory.iterdir()):
            files.append(demo.Asset(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
        profile = dataclasses.replace(original, assets=tuple(files))
        with mock.patch.dict(demo.PROFILES, {"cpu": profile}):
            yield root, directory, profile


def refresh_asset(directory, name):
    profile = demo.PROFILES["cpu"]
    path = directory / name
    asset = demo.Asset(name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    demo.PROFILES["cpu"] = dataclasses.replace(
        profile, assets=tuple(asset if value.name == name else value for value in profile.assets)
    )


class TestPhi3Assets(unittest.TestCase):
    def test_existing_full_build_runner_executes_only_offline_demo_fixtures(self):
        root = _PATH.resolve().parents[2]
        path = root / "tools/ci_build/build.py"
        spec = importlib.util.spec_from_file_location("m14_build_test", path)
        assert spec is not None and spec.loader is not None
        build = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(path.parent), *sys.path]):
            spec.loader.exec_module(build)

        class ReachedOfflineTestsError(Exception):
            pass

        reached = []
        linux = True

        def capture(command, **_kwargs):
            if command[-1].endswith("onnxruntime_test_python_phi3_demo.py"):
                self.assertEqual(
                    command[-1], str(root / "onnxruntime/test/python/onnxruntime_test_python_phi3_demo.py")
                )
                reached.append("phi3")
            elif command[-1].endswith("onnxruntime_test_python_sd15_demo.py"):
                self.assertEqual(
                    command[-1], str(root / "onnxruntime/test/python/onnxruntime_test_python_sd15_demo.py")
                )
                self.assertEqual(reached, ["phi3"])
                reached.append("sd15")
            elif command[-1].endswith("onnxruntime_test_python_stateful_qualification.py"):
                self.assertTrue(linux)
                self.assertEqual(
                    command[-1], str(root / "onnxruntime/test/python/onnxruntime_test_python_stateful_qualification.py")
                )
                self.assertEqual(reached, ["phi3", "sd15"])
                raise ReachedOfflineTestsError
            elif command[-1].endswith("onnxruntime_test_python_fpa_intb_tune.py"):
                self.assertFalse(linux)
                self.assertEqual(reached, ["phi3", "sd15"])
                raise ReachedOfflineTestsError

        args = SimpleNamespace(
            android=False,
            use_tensorrt=False,
            use_nv_tensorrt_rtx=False,
            build_shared_lib=False,
            enable_pybind=True,
            include_ops_by_config=None,
            minimal_build=None,
            skip_pip_install=True,
        )
        with (
            mock.patch.object(build, "is_macOS", return_value=False),
            mock.patch.object(build, "is_windows", return_value=False),
            mock.patch.object(build, "is_linux", side_effect=lambda: linux),
            mock.patch.object(build, "run_subprocess", side_effect=capture),
        ):
            for linux in (True, False):
                reached.clear()
                with self.subTest(linux=linux), self.assertRaises(ReachedOfflineTestsError):
                    build.run_onnxruntime_tests(args, str(root), None, "/unused-build", ("Release",))

    def test_explicit_download_and_verified_cache(self):
        with toy_assets() as (root, directory, profile):
            with mock.patch.object(demo.asset_io, "urlopen") as download:
                hashes = demo.acquire_assets(root, "cpu")
                download.assert_not_called()
            self.assertEqual(len(hashes), len(profile.assets))
            expected = (directory / "LICENSE").read_bytes()
            (directory / "LICENSE").unlink()
            response = mock.MagicMock()
            response.__enter__.return_value = io.BytesIO(expected)
            with mock.patch.object(demo.asset_io, "urlopen", return_value=response) as download:
                demo.acquire_assets(root, "cpu")
            self.assertEqual((directory / "LICENSE").read_bytes(), expected)
            self.assertIn(demo.REVISION, download.call_args.args[0])
            self.assertFalse(any(directory.glob(".download-*")))

    def test_acquisition_errors_are_not_success_or_overwrites(self):
        for mode in ("short", "long", "network", "replace"):
            with self.subTest(mode=mode), toy_assets() as (root, directory, _):
                expected = (directory / "LICENSE").read_bytes()
                (directory / "LICENSE").unlink()
                response = mock.MagicMock()
                response.__enter__.return_value = io.BytesIO(
                    expected[:-1] if mode == "short" else expected + b"x" if mode == "long" else expected
                )
                with mock.patch.object(demo.asset_io, "urlopen", return_value=response) as download:
                    if mode == "network":
                        download.side_effect = URLError("offline")
                    patch = (
                        mock.patch.object(Path, "replace", side_effect=OSError("cannot publish"))
                        if mode == "replace"
                        else mock.patch.object(demo, "REVISION", demo.REVISION)
                    )
                    with patch, self.assertRaises((ValueError, URLError, OSError)):
                        demo.acquire_assets(root, "cpu")
                self.assertFalse((directory / "LICENSE").exists())
                self.assertFalse(any(directory.glob(".download-*")))
        with toy_assets() as (root, directory, _):
            (directory / "LICENSE").write_bytes(b"invalid")
            with mock.patch.object(demo.asset_io, "urlopen") as download, self.assertRaises(ValueError):
                demo.acquire_assets(root, "cpu")
            download.assert_not_called()
            self.assertEqual((directory / "LICENSE").read_bytes(), b"invalid")

    def test_asset_hashes_git_blobs_and_argument_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blob"
            path.write_bytes(b"abc")
            git = hashlib.sha1(b"blob 3\0abc", usedforsecurity=False).hexdigest()
            demo.verify_asset(path, demo.Asset("blob", 3, git, True))
            with self.assertRaisesRegex(ValueError, "Checksum"):
                demo.verify_asset(path, demo.Asset("blob", 3, "0" * 64))
        for value in ("other", [], None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                demo.variant_directory(".", value)
        for value in (0, -1, True, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                demo._positive(value, "count")
        for pattern in ("x", "%d%d", "%d%s", None):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                demo._name(pattern, 0)

    def test_actual_interface_inspection_and_unsupported_shapes(self):
        with toy_assets() as (root, _, _):
            report = demo.inspect_assets(root, "cpu")
            self.assertEqual(len(report["caches"]), 4)
            self.assertEqual(report["cache_dtype"], "FLOAT")
        for tensor, expected in (
            ({"dtype": "FLOAT16", "shape": [1, "S"]}, ("FLOAT", (1, None))),
            ({"dtype": "FLOAT", "shape": [1]}, ("FLOAT", (1, None))),
            ({"dtype": "FLOAT", "shape": [1, 3]}, ("FLOAT", (1, None))),
            ({"dtype": "FLOAT", "shape": [2, "S"]}, ("FLOAT", (1, None))),
        ):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                demo._check_tensor(tensor, *expected)

    def test_graph_and_metadata_mismatches_are_explicit(self):
        cases = {
            "eos": "EOS",
            "filename": "filename",
            "extra_input": "ports",
            "external_name": "reference",
            "external_range": "range",
            "no_external": "external weight",
        }
        for case, message in cases.items():
            with self.subTest(case=case), toy_assets() as (root, directory, profile):
                config_path = directory / "genai_config.json"
                if case in ("eos", "filename"):
                    config = json.loads(config_path.read_text())
                    if case == "eos":
                        config["model"]["eos_token_id"] = [4]
                    else:
                        config["model"]["decoder"]["filename"] = "other.onnx"
                    config_path.write_text(json.dumps(config))
                    refresh_asset(directory, config_path.name)
                else:
                    path = directory / profile.filename
                    model = onnx.load(path, load_external_data=False)
                    if case == "extra_input":
                        model.graph.input.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, []))
                    elif case == "external_name":
                        next(
                            value for value in model.graph.initializer[0].external_data if value.key == "location"
                        ).value = "../outside"
                    elif case == "external_range":
                        next(
                            value for value in model.graph.initializer[0].external_data if value.key == "length"
                        ).value = "999999"
                    else:
                        tensor = model.graph.initializer[0]
                        tensor.data_location = TensorProto.DEFAULT
                        tensor.ClearField("external_data")
                        tensor.raw_data = (directory / (profile.filename + ".data")).read_bytes()
                    onnx.save_model(model, path)
                    refresh_asset(directory, path.name)
                with self.assertRaisesRegex(ValueError, message):
                    demo.inspect_assets(root, "cpu")


class TestPhi3Execution(unittest.TestCase):
    def test_adapter_failures_and_native_cancel_reset_cleanup_remain_explicit(self):
        with toy_assets() as (root, _, _):
            package = demo.build_package(root, "cpu", max_new_tokens=3)
            for prompt, raw in (("", False), (None, False), ("prompt", 1)):
                with self.subTest(prompt=prompt, raw=raw), self.assertRaises((ValueError, TypeError)):
                    demo.create_execution(package, prompt, FakeAdapter(), raw=raw)
            broken = mock.Mock(spec=FakeAdapter)
            broken.encode.side_effect = ValueError("adapter encode")
            with self.assertRaisesRegex(ValueError, "adapter encode"):
                demo.generate(package, "prompt", broken)
            broken.encode.side_effect = None
            broken.encode.return_value = [0, 1]
            broken.decode.side_effect = ValueError("adapter decode")
            with self.assertRaisesRegex(ValueError, "adapter decode"):
                demo.generate(package, "prompt", broken)
            execution, _, _ = demo.create_execution(package, "prompt", FakeAdapter())
            with execution:
                execution.cancel()
                with self.assertRaises(demo.ort.capi.onnxruntime_pybind11_state.Fail):
                    execution.run()
                self.assertTrue(execution.requires_reset())
                execution.reset()
                with execution.run() as result:
                    value = result.get_outputs()["tokens"]
            np.testing.assert_array_equal(value.numpy(), [2, 3])
            value.close()

    def test_provider_placement_requires_real_compute_on_the_requested_ep(self):
        def node(kind):
            return SimpleNamespace(name=kind, op_type=kind)

        def session(gpu, cpu):
            groups = [
                SimpleNamespace(ep_name="CUDAExecutionProvider", get_nodes=lambda: [node(kind) for kind in gpu]),
                SimpleNamespace(ep_name="CPUExecutionProvider", get_nodes=lambda: [node(kind) for kind in cpu]),
            ]
            return SimpleNamespace(get_provider_graph_assignment_info=lambda: groups)

        self.assertEqual(
            len(demo.provider_placement(session(["GroupQueryAttention", "MatMul"], ["Cast"]), "cuda", 1)), 2
        )
        for gpu, cpu, target, message in (
            (["MatMul"], [], "cuda", "attention layer"),
            (["GroupQueryAttention"], [], "cuda", "matrix compute"),
            (["GroupQueryAttention", "MatMul"], ["MatMulNBits"], "cuda", "CPU fallback"),
            (["MatMul"], [], "cpu", "CPU-only"),
        ):
            with self.subTest(target=target, gpu=gpu), self.assertRaisesRegex(AssertionError, message):
                demo.provider_placement(session(gpu, cpu), target, 1)

    def test_reference_rejects_bad_native_outputs_progress_and_accounting(self):
        original_step = demo.sf.StatefulExecution.step
        cases = {
            "outputs": "output collections",
            "prefill_token": "before prefill",
            "token": "greedy token mismatch",
            "incomplete": "did not complete",
            "invocations": "invocation accounting",
        }
        for case, message in cases.items():
            with self.subTest(case=case), toy_assets() as (root, _, _):
                package = demo.build_package(root, "cpu", max_new_tokens=3)
                previous = None

                def changed_step(execution, *, case=case):
                    nonlocal previous
                    if case == "incomplete" and previous is not None and previous.completed:
                        proxy = mock.MagicMock()
                        proxy.__enter__.return_value = proxy
                        proxy.__exit__.return_value = False
                        proxy.get_info.return_value = dataclasses.replace(
                            previous, executed=False, completed=False, generated_token=None
                        )
                        return proxy
                    result = original_step(execution)
                    info = result.get_info()
                    previous = info
                    proxy = mock.MagicMock(wraps=result)
                    proxy.__enter__.return_value = proxy
                    proxy.__exit__.side_effect = result.__exit__
                    if case == "outputs" and info.executed:
                        proxy.get_outputs.return_value = {}
                    elif case == "prefill_token" and info.component_name == "prefill":
                        info = dataclasses.replace(info, executed=False, generated_token=1)
                    elif case == "token" and info.generated_token is not None:
                        info = dataclasses.replace(info, generated_token=0)
                    elif case == "incomplete":
                        info = dataclasses.replace(info, completed=False)
                    elif case == "invocations" and info.completed:
                        info = dataclasses.replace(info, invocations=999)
                    proxy.get_info.return_value = info
                    return proxy

                with (
                    mock.patch.object(demo.sf.StatefulExecution, "step", new=changed_step),
                    self.assertRaisesRegex(AssertionError, message),
                ):
                    demo.validate_generation(package, "prompt", FakeAdapter())

    def test_reference_rejects_missing_caches_and_wrong_replay_counts(self):
        original_state = demo.sf.StatefulExecution.get_state
        original_run = demo.sf.StatefulExecution.run
        for stage, message in ((1, "prefill cache snapshot"), (2, "cache length/device")):
            with self.subTest(stage=stage), toy_assets() as (root, _, _):
                package = demo.build_package(root, "cpu", max_new_tokens=3)
                calls = 0

                def changed_state(execution, *, stage=stage):
                    nonlocal calls
                    calls += 1
                    if calls == stage:
                        result = mock.Mock()
                        result.get_outputs.return_value = {}
                        return result
                    return original_state(execution)

                with (
                    mock.patch.object(demo.sf.StatefulExecution, "get_state", new=changed_state),
                    self.assertRaisesRegex(AssertionError, message),
                ):
                    demo.validate_generation(package, "prompt", FakeAdapter())
        with toy_assets() as (root, _, _):
            package = demo.build_package(root, "cpu", max_new_tokens=3)

            def changed_replay(execution):
                result = original_run(execution)
                proxy = mock.MagicMock(wraps=result)
                proxy.__enter__.return_value = proxy
                proxy.__exit__.side_effect = result.__exit__
                proxy.get_info.return_value = dataclasses.replace(result.get_info(), invocations=999)
                return proxy

            with (
                mock.patch.object(demo.sf.StatefulExecution, "run", new=changed_replay),
                self.assertRaisesRegex(AssertionError, "Reset/replay"),
            ):
                demo.validate_generation(package, "prompt", FakeAdapter())

    def test_stateless_reference_checkpoints_cache_snapshots_and_bootstrap(self):
        for budget, expected in ((0, []), (1, [2]), (3, [2, 3])):
            with self.subTest(budget=budget), toy_assets() as (root, _, _):
                package = demo.build_package(root, "cpu", max_new_tokens=budget)
                result = demo.validate_generation(package, "prompt", FakeAdapter())
                self.assertEqual(result["token_ids"], expected)
                self.assertEqual(result["invocations"], max(1, len(expected)))
                self.assertEqual(result["cache_count"], 4)
                self.assertEqual(result["cache_length"], 2 + max(0, len(expected) - 1))
                self.assertTrue(result["retained_cache_snapshots_unchanged"])
                self.assertTrue(result["reset_replay_verified"])
                self.assertLessEqual(result["maximum_absolute_checkpoint_error"], 1e-4)

    def test_native_packages_budgets_eos_replay_and_programmatic_inputs(self):
        for budget, expected in ((0, []), (1, [2]), (3, [2, 3])):
            with self.subTest(budget=budget), toy_assets() as (root, _, _):
                package = demo.build_package(root, "cpu", max_new_tokens=budget)
                result = demo.generate(package, "prompt", FakeAdapter())
                self.assertEqual(result["token_ids"], expected)
                self.assertEqual(result["invocations"], max(1, len(expected)))
                execution, _, length = demo.create_execution(package, "prompt", FakeAdapter())
                self.assertEqual(length, 2)
                with execution:
                    np.testing.assert_array_equal(execution.run().get_outputs()["tokens"].numpy(), expected)
                    execution.reset()
                    np.testing.assert_array_equal(execution.run().get_outputs()["tokens"].numpy(), expected)
        with toy_assets() as (root, _, _):
            package = demo.build_package(root, "cpu", max_new_tokens=3)
            result = demo.generate(package, "eos", FakeAdapter())
            self.assertEqual(result["token_ids"], [3])
            self.assertEqual(result["text"], "")
            self.assertEqual(result["invocations"], 1)

    def test_failed_validation_preserves_existing_manifest(self):
        with toy_assets() as (root, _, _):
            package = demo.build_package(root, "cpu", max_new_tokens=1)
            previous = (package / "manifest.json").read_bytes()
            with (
                mock.patch.object(demo.sf.ModelPackage, "create_chain", side_effect=ValueError("native validation")),
                self.assertRaisesRegex(ValueError, "native validation"),
            ):
                demo.build_package(root, "cpu", max_new_tokens=3)
            self.assertEqual((package / "manifest.json").read_bytes(), previous)
            self.assertFalse(any(package.glob(".validate-*")))

    def test_context_metadata_tokens_and_configuration_errors(self):
        with toy_assets() as (root, _, _):
            package = demo.build_package(root, "cpu", max_new_tokens=3)
            for tokens in ([], "bad", [True], [-1], [4], list(range(33))):
                adapter = SimpleNamespace(encode=lambda *_args, tokens=tokens, **_kwargs: tokens)
                with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                    demo.create_execution(package, "prompt", adapter)
            with self.assertRaisesRegex(ValueError, "context"):
                demo.create_execution(package, "prompt", SimpleNamespace(encode=lambda *_args, **_kwargs: [0] * 30))
            with self.assertRaisesRegex(ValueError, "context"):
                demo.build_package(root, "cpu", max_new_tokens=33)
            data = json.loads((package / "manifest.json").read_text())
            data["additional_metadata"]["phi3_demo"]["max_new_tokens"] = 4
            (package / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "budget disagree"):
                demo.package_configuration(package)
            data["additional_metadata"]["phi3_demo"]["revision"] = "other"
            (package / "manifest.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "provenance"):
                demo.package_configuration(package)
        for budget, device in ((-1, 0), (True, 0), (1, -1), (1, True)):
            with self.subTest(budget=budget, device=device), self.assertRaises(ValueError):
                demo.build_package(".", "cpu", max_new_tokens=budget, device_id=device)
        for target, device in (("cpu", 1), ("cpu", -1), ("cpu", True), ("cuda", 999)):
            with self.subTest(target=target, device=device), self.assertRaises(ValueError):
                demo.session_options(target, device)
        missing_memory = SimpleNamespace(ep_name="CUDAExecutionProvider", memory_info=lambda _kind: None)
        gpu = SimpleNamespace(ep_name="CUDAExecutionProvider", memory_info=lambda _kind: SimpleNamespace(device_id=2))
        with (
            mock.patch.object(demo.ort, "get_ep_devices", return_value=[missing_memory, gpu]),
            mock.patch.object(demo.ort, "SessionOptions") as constructor,
        ):
            result = demo.session_options("cuda", 2)
            self.assertIs(result, constructor.return_value)
            result.add_provider_for_devices.assert_called_once_with([gpu], {})

    def test_local_tokenizer_boundary_and_cli(self):
        tokenizer = SimpleNamespace(
            encode=mock.Mock(return_value=[0, 1]),
            apply_chat_template=mock.Mock(return_value=[0, 1]),
            decode=mock.Mock(return_value="text"),
        )
        factory = mock.Mock(return_value=tokenizer)
        module = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))
        with mock.patch.object(demo.importlib, "import_module", return_value=module):
            adapter = demo.LocalTokenizer("/local")
        self.assertEqual(
            factory.call_args.kwargs,
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            },
        )
        self.assertEqual(adapter.encode("prompt"), [0, 1])
        self.assertEqual(adapter.encode("prompt", raw=True), [0, 1])
        self.assertEqual(adapter.decode([2]), "text")
        for prompt, raw in (("", False), (" ", False), (None, False), ("ok", 1)):
            with self.subTest(prompt=prompt, raw=raw), self.assertRaises((TypeError, ValueError)):
                adapter.encode(prompt, raw=raw)
        with toy_assets() as (root, _, _), mock.patch.object(demo, "LocalTokenizer", return_value=FakeAdapter()):
            with redirect_stdout(io.StringIO()) as output:
                demo.main(["--assets", str(root), "--target", "cpu", "acquire"])
                demo.main(["--assets", str(root), "--target", "cpu", "inspect"])
                demo.main(["--assets", str(root), "--target", "cpu", "build", "--max-new-tokens", "3"])
                demo.main(["--assets", str(root), "--target", "cpu", "run", "--prompt", "prompt", "--raw"])
                demo.main(["--assets", str(root), "--target", "cpu", "validate", "--prompt", "prompt"])
            self.assertIn('"token_ids"', output.getvalue())
        with (
            mock.patch.object(sys, "argv", [str(_PATH), "--help"]),
            redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as stopped,
        ):
            runpy.run_path(str(_PATH), run_name="__main__")
        self.assertEqual(stopped.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
