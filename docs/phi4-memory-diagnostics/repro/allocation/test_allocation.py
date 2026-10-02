# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Small invented unit fixtures only: never a substitute for a fresh GPU replay."""

import builtins
import copy
import csv
import hashlib
import importlib
import shutil
import struct
import types
import unittest
import uuid
from unittest import mock

import allocation
import config
import monitor
import run_phase_b
import run_validation
import support as s


def example_config():
    value = s.load(s.HERE / "config.example.json")
    for variant in s.VARIANTS:
        runtime = value["runtimes"][variant]
        runtime["runtime_sha256"] = dict.fromkeys(config.LIBRARIES, "0" * 64)
        runtime["runtime_sha256"]["libonnxruntime-genai.so"] = ("1" if variant == "baseline" else "2") * 64
        for key in ("phase_a", "allocation_tests", "existing_tests"):
            runtime[key]["sha256"] = "3" * 64
    value["phase_b"]["sha256"] = "4" * 64
    return value


def summary_records():
    records = []
    for pid, (context, repetition, variant) in enumerate(s.PLAN, 1):
        factor = repetition * (1 if variant == "baseline" else 2)
        fields = {
            "total_allocated_bytes": 8388608,
            "reserved_bytes": 8388608,
            "bfc_region_bytes": 8388608,
            "bytes_in_use": 6291456,
            "bytes_requested_in_use": 4194304,
            "arena_slack_bytes": 2097152,
            "internal_fragmentation_bytes": 2097152,
            "max_bytes_in_use": 7340032,
            "max_alloc_size": 3145728,
            "num_allocs": 64,
            "num_reserves": 8,
            "num_arena_extensions": 2,
            "num_arena_shrinkages": 1,
        }
        arenas = {
            stage: {
                group: {key: value * factor for key, value in fields.items()}
                for group in ("device_sums", "pinned_sums")
            }
            for stage in s.CHECKPOINTS
        }
        records.append(
            {
                "context": context,
                "repetition": repetition,
                "variant": variant,
                "pid": pid,
                "status": "complete",
                "phase_a_output_exact_match": True,
                "interference_detected": False,
                "arenas": arenas,
                **dict.fromkeys(s.METRICS, 25 if variant == "baseline" else 40),
            }
        )
    return records


class ScratchTest(unittest.TestCase):
    def setUp(self):
        self.root = s.HERE / f".test-state-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def tokens(self, directory, tokens=None):
        directory.mkdir(exist_ok=True)
        tokens = list(range(64)) if tokens is None else tokens
        s.save(directory / "generated-ids.json", tokens)
        (directory / "generated-ids.i32le").write_bytes(struct.pack("<64i", *tokens))

    def logits(self, directory, value=1.0):
        (directory / "logits.bin").write_bytes(struct.pack("<200064f", *([value] * 200064)))
        s.save(
            directory / "logits.json",
            {"dtype": "float32", "shape": [1, 1, 200064], "byte_count": 800256, "nan_count": 0},
        )
        s.save(directory / "effective-config.json", {"unit_fixture": True})


class ContractTests(unittest.TestCase):
    def test_artifact_and_manifest_relative_path(self):
        self.assertTrue(allocation.artifact_check()["artifact_check_passed"])
        self.assertEqual(s.MANIFEST.resolve(), s.HERE.parents[1] / "evidence/model-manifest.json")
        self.assertEqual(s.COUNTS, {2048: 1801, 32768: 28747})

    def test_configuration_no_defaults_or_overlapping_output(self):
        value = example_config()
        self.assertIs(config.validate_config(value), value)
        for mutate in (
            lambda c: c.pop("gpu_uuid"),
            lambda c: c.update(output_dir=c["model_dir"]),
            lambda c: c.update(output_dir=c["model_dir"] + "/output"),
            lambda c: c.update(gpu_index=-1),
            lambda c: c["source_pins"].update(genai="HEAD"),
            lambda c: c["resource_limits"].update(min_gpu_free_bytes=0),
            lambda c: c["runtimes"]["patched"].update(genai_library_dir=c["runtimes"]["baseline"]["genai_library_dir"]),
            lambda c: c["runtimes"]["baseline"]["runtime_sha256"].update({"libonnxruntime.so": "not-a-hash"}),
        ):
            invalid = copy.deepcopy(value)
            mutate(invalid)
            with self.assertRaises(RuntimeError):
                config.validate_config(invalid)

    def test_missing_prerequisites_block_without_importing_nvml(self):
        with (
            mock.patch("importlib.import_module", side_effect=AssertionError("Unexpected external import")),
            self.assertRaisesRegex(RuntimeError, "Missing external prerequisite"),
        ):
            config.check(example_config())

    def test_imports_are_host_only(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"pynvml", "numpy", "onnx", "onnxruntime", "torch"}:
                raise AssertionError(f"Import-time external dependency: {name}")
            return original(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            for module in (s, config, monitor, run_validation, run_phase_b, allocation):
                importlib.reload(module)

    def test_input_hash_and_i32le_contract(self):
        with self.assertRaisesRegex(RuntimeError, "byte count"):
            s.validate_input(b"\0" * 10, 2048)
        raw = struct.pack("<1801i", *range(1801))
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            s.validate_input(raw, 2048)
        # A private in-memory digest override tests decoding without publishing saved inputs.
        with mock.patch.dict(s.INPUT_HASHES, {2048: hashlib.sha256(raw).hexdigest()}):
            self.assertEqual(s.validate_input(raw, 2048), tuple(range(1801)))
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                s.validate_input(raw[:-1] + b"\1", 2048)
        invalid = struct.pack("<1801i", -1, *range(1800))
        with (
            mock.patch.dict(s.INPUT_HASHES, {2048: hashlib.sha256(invalid).hexdigest()}),
            self.assertRaisesRegex(RuntimeError, "vocabulary"),
        ):
            s.validate_input(invalid, 2048)

    def test_timing_uses_63_decode_tokens_and_completion_barriers(self):
        timing = {
            "clock": "CLOCK_MONOTONIC",
            "generation_steps": 64,
            "start_ns": 1_000_000_000,
            "first_sync_start_ns": 2_990_000_000,
            "first_token_ns": 3_000_000_000,
            "final_sync_start_ns": 6_999_000_000,
            "end_ns": 7_000_000_000,
        }
        metrics = s.timing_metrics(timing)
        self.assertEqual(metrics["ttft_ms"], 2000)
        self.assertEqual(metrics["decode_tokens_per_second"], 15.75)
        self.assertEqual(metrics["first_completion_sync_ms"], 10)
        self.assertEqual(metrics["final_completion_sync_ms"], 1)
        for change in (
            {"generation_steps": 63},
            {"end_ns": timing["first_token_ns"]},
            {"clock": "WALL"},
            {"first_sync_start_ns": timing["start_ns"]},
        ):
            with self.assertRaises(RuntimeError):
                s.timing_metrics(timing | change)

    def test_complete_queries_only_and_fail_closed_interference(self):
        def sample(start, end):
            return {
                "memory_query_start_ns": start,
                "query_end_ns": end,
                "process_present": True,
                "process_used_bytes": 100,
                "foreign_pids": [],
                "process_query_errors": [],
            }

        samples = [sample(0, 11), sample(10, 12), sample(50, 52), sample(99, 101)]
        self.assertEqual(s.inference_samples(samples, {"start_ns": 10, "end_ns": 100}), samples[1:3])
        for change in (
            {"process_used_bytes": None},
            {"foreign_pids": [123]},
            {"process_query_errors": ["denied"]},
            {"process_present": False},
        ):
            invalid = copy.deepcopy(samples)
            invalid[1].update(change)
            with self.assertRaises(RuntimeError):
                s.inference_samples(invalid, {"start_ns": 10, "end_ns": 100})

    def test_resource_gates_reject_low_memory_disk_and_swap(self):
        memory = "MemAvailable: 33554432 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
        baseline = example_config()
        with (
            mock.patch("pathlib.Path.read_text", return_value=memory),
            mock.patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=2**31)),
        ):
            config.host_resources(baseline)
            for change in (
                {"min_host_available_bytes": 2**40},
                {"min_output_free_bytes": 2**40},
            ):
                value = copy.deepcopy(baseline)
                value["resource_limits"].update(change)
                with self.assertRaisesRegex(RuntimeError, "resource gate"):
                    config.host_resources(value)
        with (
            mock.patch("pathlib.Path.read_text", return_value=memory.replace("SwapTotal: 0", "SwapTotal: 1")),
            mock.patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=2**31)),
            self.assertRaisesRegex(RuntimeError, "Swap resource gate"),
        ):
            config.host_resources(baseline)

    def test_nvml_unavailable_attribution_and_query_permissions_fail_closed(self):
        nv = types.SimpleNamespace(
            NVMLError_NotSupported=NotImplementedError,
            NVMLError_NoPermission=PermissionError,
            nvmlDeviceGetMemoryInfo=lambda _: types.SimpleNamespace(total=2048, used=1024, free=1024),
            nvmlDeviceGetComputeRunningProcesses=lambda _: [types.SimpleNamespace(pid=123, usedGpuMemory=2**64 - 1)],
            nvmlDeviceGetGraphicsRunningProcesses=lambda _: [],
            nvmlDeviceGetUtilizationRates=lambda _: types.SimpleNamespace(gpu=0),
        )
        sample = monitor.gpu_sample(nv, None, 123)
        self.assertTrue(sample["process_present"])
        self.assertIsNone(sample["process_used_bytes"])
        with mock.patch.object(nv, "nvmlDeviceGetGraphicsRunningProcesses", side_effect=PermissionError("denied")):
            sample = monitor.gpu_sample(nv, None, 123)
        with self.assertRaisesRegex(RuntimeError, "Cannot monitor"):
            s.check_interference(sample)

    def test_plan_and_environment_are_symmetric(self):
        self.assertEqual(len(s.PLAN), 12)
        for index in range(0, 12, 2):
            baseline, patched = s.PLAN[index : index + 2]
            self.assertEqual(baseline[:2], patched[:2])
            self.assertEqual((baseline[2], patched[2]), s.VARIANTS)
        value = example_config()
        with mock.patch.dict("os.environ", {"LD_PRELOAD": "bad.so", "LD_AUDIT": "bad.so"}):
            environments = [config.environment(value, variant, True) for variant in s.VARIANTS]
        for env, overrides, _ in environments:
            self.assertNotIn("LD_PRELOAD", env)
            self.assertNotIn("LD_AUDIT", env)
            self.assertEqual(overrides["ORT_ARENA_DIAGNOSTICS"], "1")
            self.assertEqual(overrides["ORTGENAI_LOGITS_ALLOCATION_TRACE"], "0")
        self.assertEqual(
            {k: v for k, v in environments[0][1].items() if k != "LD_LIBRARY_PATH"},
            {k: v for k, v in environments[1][1].items() if k != "LD_LIBRARY_PATH"},
        )

    def test_arena_accounting_separates_pinned_and_rejects_shrink(self):
        lines = []
        for checkpoint in s.CHECKPOINTS:
            for name, amount in (("Cuda", 100), ("CudaPinned", 20)):
                fields = dict.fromkeys(s.ARENA_FIELDS, 0)
                fields.update(total_allocated_bytes=amount, bfc_region_bytes=amount, arena_slack_bytes=amount)
                lines.append(
                    f"[ ARENA CHECKPOINT ] checkpoint={checkpoint} allocator={name} "
                    "device_id=0 phase=snapshot " + " ".join(f"{k}={v}" for k, v in fields.items())
                )
            lines.append(f"[ ARENA SHRINK ] checkpoint={checkpoint} requested=0 reclaimed_bytes=0 arena_count=2")
        text = "\n".join(lines)
        result = s.parse_arenas(text)
        self.assertEqual(result["post_initialize"]["device_sums"]["total_allocated_bytes"], 100)
        self.assertEqual(result["post_initialize"]["pinned_sums"]["total_allocated_bytes"], 20)
        with self.assertRaises(RuntimeError):
            s.parse_arenas(text.replace("requested=0", "requested=1"))
        with self.assertRaises(RuntimeError):
            s.parse_arenas(text.replace("arena_slack_bytes=100", "arena_slack_bytes=99"))

    def test_summary_requires_full_matrix_and_signed_deltas(self):
        records = summary_records()
        result = s.summarize(records)
        expected_memory = {
            "total_allocated_bytes": {"median": 16, "min": 8, "max": 24},
            "reserved_bytes": {"median": 16, "min": 8, "max": 24},
            "bfc_region_bytes": {"median": 16, "min": 8, "max": 24},
            "bytes_in_use": {"median": 12, "min": 6, "max": 18},
            "bytes_requested_in_use": {"median": 8, "min": 4, "max": 12},
            "arena_slack_bytes": {"median": 4, "min": 2, "max": 6},
            "internal_fragmentation_bytes": {"median": 4, "min": 2, "max": 6},
            "max_bytes_in_use": {"median": 14, "min": 7, "max": 21},
            "max_alloc_size": {"median": 6, "min": 3, "max": 9},
        }
        expected_counts = {
            "num_allocs": {"median": 128, "min": 64, "max": 192},
            "num_reserves": {"median": 16, "min": 8, "max": 24},
            "num_arena_extensions": {"median": 4, "min": 2, "max": 6},
            "num_arena_shrinkages": {"median": 2, "min": 1, "max": 3},
        }
        self.assertEqual(set(s.ARENA_BYTE_FIELDS), set(expected_memory))
        self.assertEqual(set(s.ARENA_COUNT_FIELDS), set(expected_counts))
        self.assertEqual(set(s.ARENA_FIELDS), set(expected_memory) | set(expected_counts))
        for context in (2048, 32768):
            for variant, factor in (("baseline", 1), ("patched", 2)):
                group = result["groups"][f"{context}-{variant}"]
                for checkpoint in s.CHECKPOINTS:
                    for arena in ("device_sums", "pinned_sums"):
                        for section, expected in (("arenas_mib", expected_memory), ("arena_counts", expected_counts)):
                            self.assertEqual(
                                group[section][checkpoint][arena],
                                {
                                    field: {stat: value * factor for stat, value in values.items()}
                                    for field, values in expected.items()
                                },
                            )
            for checkpoint in s.CHECKPOINTS:
                self.assertEqual(
                    result["paired_baseline_minus_patched"][str(context)]["device_arena_mib"][checkpoint],
                    {
                        "total_allocated_bytes": {"median": -16, "min": -24, "max": -8},
                        "bytes_in_use": {"median": -12, "min": -18, "max": -6},
                        "arena_slack_bytes": {"median": -4, "min": -6, "max": -2},
                    },
                )
        self.assertEqual(
            result["paired_baseline_minus_patched"]["32768"]["process_inference_peak_mib"],
            {"median": -15, "min": -15, "max": -15},
        )
        for invalid in (records[:-1], records[::-1], [records[0]] * 12):
            with self.assertRaises(RuntimeError):
                s.summarize(invalid)
        records[0]["phase_a_output_exact_match"] = False
        with self.assertRaises(RuntimeError):
            s.summarize(records)

    def test_report_revalidates_counter_schema(self):
        root = s.HERE / "invented-report"
        records = summary_records()
        summary = s.summarize(records)
        files = {
            root / "status.json": {"status": "complete"},
            root / "manifest.json": {"host_check": {"runtime_sha256": {}}},
            root / "measurement-results/records.json": records,
            root / "measurement-results/summary.json": summary,
        }
        for record in records:
            name = f"context-{record['context']}-rep-{record['repetition']}-{record['variant']}"
            files[root / "measurement-results" / name / "record.json"] = record
        with (
            mock.patch.object(s, "load", side_effect=files.__getitem__),
            mock.patch.object(run_validation, "verify_phase_a"),
            mock.patch.object(run_phase_b, "analyze_capture", side_effect=records * 3),
        ):
            self.assertEqual(run_phase_b.report(root), summary)
            legacy = copy.deepcopy(summary)
            for group in legacy["groups"].values():
                counters = group.pop("arena_counts")
                for checkpoint, arenas in counters.items():
                    for arena, fields in arenas.items():
                        group["arenas_mib"][checkpoint][arena].update(
                            {
                                key: {stat: value / 1048576 for stat, value in stats.items()}
                                for key, stats in fields.items()
                            }
                        )
            files[root / "measurement-results/summary.json"] = legacy
            with self.assertRaisesRegex(RuntimeError, "Summary differs"):
                run_phase_b.report(root)
            corrupt = copy.deepcopy(summary)
            corrupt["groups"]["2048-baseline"]["arena_counts"]["post_initialize"]["device_sums"]["num_allocs"][
                "median"
            ] = 128 / 1048576
            files[root / "measurement-results/summary.json"] = corrupt
            with self.assertRaisesRegex(RuntimeError, "Summary differs"):
                run_phase_b.report(root)


class FileGateTests(ScratchTest):
    def runtime_fixture(self):
        value = example_config()
        ort = self.root / "ort/lib"
        ort.mkdir(parents=True)
        value["ort_home"] = str(ort.parent)
        for variant in s.VARIANTS:
            runtime = value["runtimes"][variant]
            build = self.root / variant
            build.mkdir()
            runtime["genai_library_dir"] = str(build)
            for name in config.LIBRARIES:
                path = (build if name.startswith("libonnxruntime-genai") else ort) / name
                data = (name + (variant if name.startswith("libonnxruntime-genai") else "")).encode()
                path.write_bytes(data)
                runtime["runtime_sha256"][name] = hashlib.sha256(data).hexdigest()
            for key, name in (("allocation_tests", "logits_allocation_tests"), ("existing_tests", "unit_tests")):
                path = build / name
                path.write_bytes(b"Invented host fixture, not a native executable")
                path.chmod(0o700)
                runtime[key] = {"path": str(path), "sha256": s.sha256(path)}
        return value

    def test_test_build_layout_and_adjacent_companion_hashes(self):
        value = self.runtime_fixture()
        with (
            mock.patch.object(s, "HERE", self.root / "artifact"),
            mock.patch("subprocess.check_output", side_effect=AssertionError("Native execution not permitted")),
        ):
            self.assertIs(config.validate_config(value), value)
            for variant in s.VARIANTS:
                runtime = value["runtimes"][variant]
                hashes = config.runtime_hashes(value, variant)
                companion = str(self.root / variant / "libonnxruntime-genai-cuda.so")
                self.assertEqual(hashes[companion], runtime["runtime_sha256"]["libonnxruntime-genai-cuda.so"])
                for key in ("allocation_tests", "existing_tests"):
                    config.verify_native(runtime[key], "tests")

    def test_detached_or_wrong_variant_tests_block_before_gpu(self):
        value = self.runtime_fixture()
        detached = self.root / "detached"
        detached.mkdir()
        with (
            mock.patch.object(s, "HERE", self.root / "artifact"),
            mock.patch.object(allocation, "artifact_check"),
            mock.patch.object(config, "runtime_hashes") as hashes,
            mock.patch.object(monitor, "launch", side_effect=AssertionError("Native execution not permitted")),
        ):
            for variant in s.VARIANTS:
                other = "patched" if variant == "baseline" else "baseline"
                for key in ("allocation_tests", "existing_tests"):
                    for destination in (detached, self.root / other):
                        with self.subTest(variant=variant, key=key, destination=destination):
                            invalid = copy.deepcopy(value)
                            source = value["runtimes"][variant][key]["path"]
                            path = destination / source.rsplit("/", 1)[-1]
                            if destination == detached:
                                path.write_bytes(b"Invented detached executable")
                            invalid["runtimes"][variant][key]["path"] = str(path)
                            with self.assertRaisesRegex(RuntimeError, f"{key} must remain"):
                                allocation.execute(invalid)
            hashes.assert_not_called()

    def test_missing_or_wrong_adjacent_companion_rejected(self):
        value = self.runtime_fixture()
        for variant in s.VARIANTS:
            path = self.root / variant / "libonnxruntime-genai-cuda.so"
            original = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                config.runtime_hashes(value, variant)
            path.write_bytes(b"Wrong CUDA companion")
            with self.assertRaisesRegex(RuntimeError, "Runtime hash changed"):
                config.runtime_hashes(value, variant)
            path.write_bytes(original)
            config.runtime_hashes(value, variant)

    def test_report_files_preserve_raw_bytes_and_counts(self):
        records = summary_records()
        summary = run_phase_b.write_reports(self.root, records)
        self.assertEqual(s.load(self.root / "summary.json"), summary)
        with (self.root / "arenas.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 72)
        self.assertEqual(rows[0]["total_allocated_bytes"], "8388608")
        self.assertEqual(rows[0]["num_allocs"], "64")
        self.assertEqual(rows[0]["num_reserves"], "8")
        self.assertEqual(rows[0]["num_arena_extensions"], "2")
        self.assertEqual(rows[0]["num_arena_shrinkages"], "1")
        group = summary["groups"]["2048-baseline"]
        self.assertEqual(
            group["arenas_mib"]["post_initialize"]["device_sums"]["total_allocated_bytes"],
            {"median": 16, "min": 8, "max": 24},
        )
        self.assertEqual(
            group["arena_counts"]["post_initialize"]["device_sums"]["num_allocs"],
            {"median": 128, "min": 64, "max": 192},
        )

    def test_fixture_identity_rejects_missing_corrupt_and_pointer_files(self):
        path = self.root / "tiny-model.onnx"
        path.write_bytes(b"unit")
        fixture = {
            "source_revision": s.GENAI_COMMIT,
            "files": [{"path": path.name, "bytes": 4, "sha256": hashlib.sha256(b"unit").hexdigest()}],
        }
        with (
            mock.patch.object(config, "absolute_path", return_value=self.root),
            mock.patch.object(s, "load", return_value=fixture),
        ):
            self.assertTrue(config.verify_fixtures(self.root)["fixture_check_passed"])
            path.write_bytes(b"edit")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                config.verify_fixtures(self.root)
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1")
            with self.assertRaisesRegex(RuntimeError, "Missing/incomplete"):
                config.verify_fixtures(self.root)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing/incomplete"):
                config.verify_fixtures(self.root)

    def test_phase_a_gates_full_logits_and_64_tokens(self):
        left, right = self.root / "left", self.root / "right"
        for directory in (left, right):
            self.tokens(directory)
            self.logits(directory)
        self.assertTrue(s.compare_phase_a(left, right)["logits_exact_bytes_equal"])
        self.logits(right, 2.0)
        with self.assertRaisesRegex(RuntimeError, "Phase B prohibited"):
            s.compare_phase_a(left, right)
        self.logits(right)
        self.tokens(right, [999, *range(1, 64)])
        with self.assertRaisesRegex(RuntimeError, "Phase B prohibited"):
            s.compare_phase_a(left, right)

    def test_reject_nan_and_truncated_logits(self):
        self.tokens(self.root)
        self.logits(self.root, float("nan"))
        with self.assertRaises(RuntimeError):
            s.verify_logits(self.root)
        (self.root / "logits.bin").write_bytes(b"\0")
        with self.assertRaises(RuntimeError):
            s.verify_logits(self.root)

    def test_maps_reject_mixed_variant_and_changed_runtime(self):
        expected = {}
        for name in config.LIBRARIES:
            path = self.root / name
            path.write_bytes(name.encode())
            expected[str(path)] = s.sha256(path)
        maps = self.root / "maps.txt"
        maps.write_text("".join(f"0000-1000 r-xp 0 00:00 0 {path}\n" for path in expected))
        self.assertEqual(s.verify_maps(maps, expected), expected)
        original = maps.read_text()
        maps.write_text(original.replace("libonnxruntime-genai-cuda.so", "other/libonnxruntime-genai-cuda.so"))
        with self.assertRaises(RuntimeError):
            s.verify_maps(maps, expected)
        maps.write_text(original)
        (self.root / "libonnxruntime-genai.so").write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            s.verify_maps(maps, expected)

    def test_regression_expected_failure_is_not_arbitrary_failure(self):
        xml = self.root / "gtest.xml"
        cases = [
            f'<testcase name="NoPromptSizedFp32/{i}">'
            + ("<failure>UNWANTED_PROMPT_SIZED_FP32</failure>" if i < 6 else "")
            + "</testcase>"
            for i in range(9)
        ]
        xml.write_text("<testsuite>" + "".join(cases) + "</testsuite>")
        self.assertTrue(s.test_result(xml, 1, 9, True)["expected_failure_confirmed"])
        xml.write_text(xml.read_text().replace("UNWANTED_PROMPT_SIZED_FP32", "OOM"))
        with self.assertRaises(RuntimeError):
            s.test_result(xml, 1, 9, True)
        xml.write_text('<testsuite><testcase name="x"><skipped/></testcase></testsuite>')
        with self.assertRaises(RuntimeError):
            s.test_result(xml, 0, 1)

    def test_report_rejects_incomplete_or_fabricated_gate(self):
        s.save(self.root / "status.json", {"status": "failed"})
        with self.assertRaises(RuntimeError):
            run_phase_b.report(self.root)
        s.save(self.root / "status.json", {"status": "complete"})
        with self.assertRaises(FileNotFoundError):
            run_phase_b.report(self.root)
        (self.root / "phase-a").mkdir()
        s.save(self.root / "phase-a/complete.json", {"captures": 4, "phase_b_runs": 0, "runtime_sha256": {}})
        with self.assertRaises(FileNotFoundError):
            run_validation.verify_phase_a(self.root / "phase-a", {})


if __name__ == "__main__":
    unittest.main()
