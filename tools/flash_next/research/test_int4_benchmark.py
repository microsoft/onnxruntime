import types
import unittest
from unittest.mock import patch

import benchmark_webgpu_int4 as benchmark


class Params:
    def __init__(self, model):
        self.model = model

    def set_search_options(self, **values):
        self.model.options = values


class Generator:
    def __init__(self, model, params):
        self.model = model
        self.tokens = []

    def append_tokens(self, prompt):
        self.prompt = list(prompt)

    def is_done(self):
        return len(self.tokens) >= self.model.limit

    def generate_next_token(self):
        self.tokens.append(len(self.tokens) + 1)

    def get_next_tokens(self):
        return [self.tokens[-1]]

    def get_sequence(self, _):
        return self.prompt + self.tokens + ([99] if self.model.corrupt else [])

    def set_runtime_option(self, key, value):
        self.model.profile_calls.append((len(self.tokens), key, value))


class RequestTest(unittest.TestCase):
    def invoke(self, limit=4, corrupt=False, profile_after=None, on_decode_step=None):
        model = types.SimpleNamespace(limit=limit, corrupt=corrupt, profile_calls=[])
        og = types.SimpleNamespace(GeneratorParams=Params, Generator=Generator)
        tokenizer = types.SimpleNamespace(decode=lambda tokens: str(tokens))
        with patch.object(benchmark.time, "perf_counter", side_effect=[0., 1., 2., 3., 4.]):
            profile_options = {} if profile_after is None else {
                "decode_profile_prefix": "decode-profile", "profile_after_tokens": profile_after}
            result = benchmark.run_request(og, model, tokenizer, [10, 20], 4, 16,
                                           on_decode_step=on_decode_step, **profile_options)
        return result, model

    def test_exact_output_count_and_first_to_last_timing(self):
        result, model = self.invoke()
        self.assertEqual(result["output_token_ids"], [1, 2, 3, 4])
        self.assertEqual(result["output_tokens"], 4)
        self.assertEqual(result["ttft_seconds"], 1.)
        self.assertEqual(result["request_seconds"], 4.)
        self.assertEqual(result["decode_seconds"], 3.)
        self.assertEqual(result["decode_tps"], 1.)
        self.assertEqual(model.options, {"max_length": 6, "min_length": 6, "do_sample": False, "chunk_size": 16})
        self.assertEqual(model.profile_calls, [])

    def test_decode_profile_excludes_prefill_and_warmup_tokens(self):
        result, model = self.invoke(profile_after=2)
        self.assertEqual(model.profile_calls, [(2, "enable_profiling", "decode-profile")])
        self.assertEqual(result["profile_after_tokens"], 2)
        self.assertEqual(result["profiled_decode_tokens"], 2)

    def test_resource_usage_separates_prefill_from_decode_without_changing_timing(self):
        snapshots = [
            types.SimpleNamespace(ru_majflt=a, ru_minflt=b, ru_utime=c, ru_stime=d,
                                  ru_nvcsw=e, ru_nivcsw=f)
            for a, b, c, d, e, f in ((10, 20, 1., 2., 30, 40),
                                     (13, 25, 2., 4., 37, 49),
                                     (15, 29, 5., 8., 48, 62))
        ]
        with patch.object(benchmark.resource, "getrusage", side_effect=snapshots) as usage:
            result, _ = self.invoke()
        self.assertEqual(usage.call_count, 3)
        self.assertEqual(result["decode_tps"], 1.)
        self.assertEqual(result["process_resource_usage"]["prefill"], {
            "major_page_faults": 3, "minor_page_faults": 5,
            "user_cpu_seconds": 1., "system_cpu_seconds": 2.,
            "voluntary_context_switches": 7, "involuntary_context_switches": 9})
        self.assertEqual(result["process_resource_usage"]["decode"], {
            "major_page_faults": 2, "minor_page_faults": 4,
            "user_cpu_seconds": 3., "system_cpu_seconds": 4.,
            "voluntary_context_switches": 11, "involuntary_context_switches": 13})

    def test_step_observer_receives_completed_step_timestamps_without_changing_results(self):
        observations = []
        result, _ = self.invoke(on_decode_step=lambda index, timestamp: observations.append((index, timestamp)))
        self.assertEqual(observations, [(0, 1.), (1, 2.), (2, 3.), (3, 4.)])
        self.assertEqual(result["decode_tps"], 1.)

    def test_step_observer_errors_are_not_hidden(self):
        def fail(index, timestamp):
            raise RuntimeError("observer failed")

        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            self.invoke(on_decode_step=fail)

    def test_invalid_qsa_window_protocol_is_rejected_before_model_load(self):
        for comparison in ("--compare-qsa-rank-windows", "--compare-qsa-dense-map-windows"):
            for options in (
                ["--repetitions", "1", "--tokens", "288"],
                ["--repetitions", "2", "--tokens", "289"],
                ["--repetitions", "2", "--tokens", "288", "--compare-contexts", "2048"]):
                argv = ["benchmark", "--plugin", "candidate.so", "--output", "unused.json",
                        comparison, *options]
                with (self.subTest(comparison=comparison, options=options),
                      patch.dict(sys.modules, {"onnxruntime_genai": types.SimpleNamespace()}),
                      patch.object(sys, "argv", argv), patch("sys.stderr")):
                    with self.assertRaises(SystemExit) as error:
                        benchmark.main()
                    self.assertEqual(error.exception.code, 2)

    def test_decode_profile_rejects_prefill_and_empty_capture(self):
        for after in (0, 4):
            with self.subTest(after=after), self.assertRaisesRegex(ValueError, "Decode profiling"):
                self.invoke(profile_after=after)
        with self.assertRaisesRegex(ValueError, "prefix is required"):
            benchmark.run_request(None, None, None, [], 4, 16, profile_after_tokens=1)

    def test_invalid_decode_profiling_cli_options_are_rejected(self):
        for options in (
                ["--profile-decode-after", "2"],
                ["--profile-prefix", "profile", "--profile-decode-after", "0"],
                ["--profile-prefix", "profile", "--profile-decode-after", "4"],
                ["--profile-prefix", "profile", "--profile-decode-after", "2", "--compare-uniform-arena"]):
            argv = ["benchmark", "--plugin", "candidate.so", "--output", "unused.json",
                    "--tokens", "4", *options]
            with (self.subTest(options=options),
                  patch.dict(sys.modules, {"onnxruntime_genai": types.SimpleNamespace()}),
                  patch.object(sys, "argv", argv), patch("sys.stderr")):
                with self.assertRaises(SystemExit) as error:
                    benchmark.main()
                self.assertEqual(error.exception.code, 2)

    def test_early_completion_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "ended early at 2/4"):
            self.invoke(limit=2)

    def test_sequence_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "sequence does not match"):
            self.invoke(corrupt=True)

    def test_cuda_cannot_be_mislabeled_as_webgpu_experiment_comparison(self):
        argv = ["benchmark", "--plugin", "cuda.so", "--output", "unused.json",
                "--provider", "cuda", "--compare-experiments"]
        with (patch.dict(sys.modules, {"onnxruntime_genai": types.SimpleNamespace()}),
              patch.object(sys, "argv", argv), patch("sys.stderr")):
            with self.assertRaises(SystemExit) as error:
                benchmark.main()
        self.assertEqual(error.exception.code, 2)

    def invoke_comparison(self, cooperative, tail="0", upload="0", scoped=False, readback="0", qsa="0",
                          skinny=False, uniform_arena=False, arena="0", configured_skinny="0",
                          attention_split=False, configured_split="0", subgroup="0",
                          qsa_rank=False, configured_rank="0", gemv_n96=False, configured_dense_map="0",
                          validation_mode=None, max_pending=None, subgroup_gate=False,
                          configured_subgroup_gate="0", configured_gemv_n96="0", bind_group_cache=False,
                          configured_bind_group_cache="0", dynamic_uniforms="0", dynamic_uniform_cache=False,
                          hyper_skinny=False, configured_hyper_skinny="0"):
        payloads = {"alice-natural-pool.txt": b"text", "tokenizer.json": b"tokenizer",
                    "candidate.so": b"plugin", "model.onnx": b"graph", "genai_config.json": b"config",
                    "libonnxruntime-genai.so": b"genai", "libonnxruntime.so.1.32.0": b"ort"}
        manifest = {
            "pool_sha256": hashlib.sha256(b"text").hexdigest(),
            "tokenizer_sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            "pool_tokens": 3, "tokens_sha256": benchmark.token_hash([1, 2, 3]),
        }
        flags = []
        attention_flags = []
        projection_flags = []
        subgroup_gate_flags = []
        bind_group_cache_values = []
        hyper_skinny_values = []
        provider_overrides = []

        def request(*args):
            with output.open() as checkpoint:
                self.assertGreaterEqual(json.load(checkpoint)["load_seconds"], 0)
            pair = (os.environ["ORT_WEBGPU_QMOE_INT4_PREFILL"],
                    os.environ["ORT_WEBGPU_QMOE_INT4_COOPERATIVE"])
            flags.append(pair)
            attention_flags.append((os.environ["ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP"],
                                    os.environ["ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV"]))
            projection_flags.append((os.environ["ORT_WEBGPU_MATMUL_GEMV_N96"],
                                     os.environ["ORT_WEBGPU_QSA_PARALLEL_RANK"],
                                     os.environ["ORT_WEBGPU_QSA_DENSE_MAP"]))
            subgroup_gate_flags.append(os.environ["ORT_WEBGPU_QMOE_SUBGROUP_GATE"])
            bind_group_cache_values.append(os.environ["ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES"])
            hyper_skinny_values.append(os.environ["ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY"])
            return {"output_token_ids": [1, 3 if pair == ("1", "1") else 2]}

        og = types.SimpleNamespace(
            Tokenizer=lambda _: types.SimpleNamespace(encode=lambda _: [1, 2, 3]),
            Config=lambda _: types.SimpleNamespace(
                set_provider_option=lambda *values: provider_overrides.append(values)),
            Model=lambda _: object(),
            register_execution_provider_library=lambda *args: None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            argv = ["benchmark", "--plugin", "candidate.so", "--model", temporary, "--output", str(output),
                    "--contexts", *(["2", "3"] if scoped else ["2"]), "--tokens", "2",
                    "--repetitions", "2", "--compare-dynamic-uniform-cache" if dynamic_uniform_cache else
                    "--compare-hyper-skinny" if hyper_skinny else
                    "--compare-bind-group-cache" if bind_group_cache else
                    "--compare-subgroup-gate" if subgroup_gate else
                    "--compare-gemv-n96" if gemv_n96 else
                    "--compare-qsa-rank" if qsa_rank else
                    "--compare-attention-split" if attention_split else
                    "--compare-uniform-arena" if uniform_arena else
                    "--compare-skinny" if skinny else "--compare-experiments"]
            if scoped:
                argv.extend(["--compare-contexts", "2"])
            if validation_mode is not None:
                argv.extend(["--webgpu-validation-mode", validation_mode])
            if max_pending is not None:
                argv.extend(["--webgpu-max-pending-dispatches", str(max_pending)])
            with (
                patch.dict(sys.modules, {"onnxruntime_genai": og}),
                patch.dict(os.environ, {"ORT_WEBGPU_QMOE_INT4_COOPERATIVE": cooperative,
                                        "ORT_WEBGPU_QMOE_INT4_PREFILL": "1",
                                        "ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL": tail,
                                        "ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER": upload,
                                        "ORT_WEBGPU_SMALL_READBACK_CACHE": readback,
                                        "ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE": qsa,
                                        "ORT_WEBGPU_UNIFORM_ARENA": arena,
                                        "ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SUBGROUP": subgroup,
                                        "ORT_WEBGPU_DYNAMIC_SPARSE_ATTENTION_SPLIT_KV": configured_split,
                                        "ORT_WEBGPU_QSA_PARALLEL_RANK": configured_rank,
                                        "ORT_WEBGPU_QSA_DENSE_MAP": configured_dense_map,
                                        "ORT_WEBGPU_MATMUL_GEMV_N96": configured_gemv_n96,
                                        "ORT_WEBGPU_QMOE_SUBGROUP_GATE": configured_subgroup_gate,
                                        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES": configured_bind_group_cache,
                                        "ORT_WEBGPU_DYNAMIC_UNIFORMS": dynamic_uniforms,
                                        "ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY": configured_hyper_skinny,
                                        "ORT_WEBGPU_MATMUL_NBITS_SKINNY": configured_skinny}),
                patch.object(sys, "argv", argv),
                patch.object(Path, "read_bytes", lambda path: payloads[path.name]),
                patch.object(Path, "read_text", lambda path: (
                    "0000-ffff r-xp 0 00:00 0 /runtime/libonnxruntime-genai.so\n"
                    "1000-ffff r-xp 0 00:00 0 /runtime/libonnxruntime.so.1.32.0"
                    if str(path) == "/proc/self/maps" else json.dumps(manifest))),
                patch.object(benchmark, "natural_text_prompt", lambda pool, context, seed: ([seed, seed + 1], seed)),
                patch.object(benchmark, "run_request", side_effect=request),
                patch.object(benchmark, "summarize", side_effect=len),
                patch("builtins.print"),
            ):
                benchmark.main()
            result = json.loads(output.read_text())
        expected_flags = [("1", cooperative), ("0", "0"), ("0", "0"),
                          ("1", cooperative), ("1", cooperative), ("0", "0")]
        if (skinny or uniform_arena or attention_split or qsa_rank or gemv_n96 or subgroup_gate or
                bind_group_cache or dynamic_uniform_cache or hyper_skinny):
            expected_flags = [("1", cooperative)] * 6
        if scoped:
            expected_flags.extend([("1", cooperative)] * 3)
        self.assertEqual(flags, expected_flags)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["quality_qualified"])
        self.assertEqual(result["genai_native"]["sha256"], hashlib.sha256(b"genai").hexdigest())
        self.assertEqual(result["ort_native"]["sha256"], hashlib.sha256(b"ort").hexdigest())
        expected_summary = {"2:control": 2, "2:candidate": 2}
        if scoped:
            expected_summary["3:configured"] = 2
        self.assertEqual(result["summary"], expected_summary)
        self.assertEqual(sum(sample["warmup"] for sample in result["samples"]), 3 if scoped else 2)
        for sample in result["samples"]:
            candidate = (skinny or uniform_arena or attention_split or qsa_rank or gemv_n96 or
                         subgroup_gate or bind_group_cache or dynamic_uniform_cache or hyper_skinny or
                         sample["mode"] != "control")
            self.assertEqual(sample["matmul_n_tail_flag"], tail if candidate else "0")
            self.assertEqual(sample["small_upload_write_buffer_flag"], upload if candidate else "0")
            self.assertEqual(sample["small_readback_cache_flag"], readback if candidate else "0")
            self.assertEqual(sample["qsa_global_score_cache_flag"], qsa if candidate else "0")
            expected_dynamic = (
                {"control": "0", "candidate": "1"}.get(sample["mode"], dynamic_uniforms)
                if dynamic_uniform_cache else dynamic_uniforms if candidate else "0")
            self.assertEqual(sample["dynamic_uniforms_flag"], expected_dynamic)
            expected_hyper = (
                {"control": "0", "candidate": "1"}.get(sample["mode"], configured_hyper_skinny)
                if hyper_skinny else configured_hyper_skinny if candidate else "0")
            self.assertEqual(sample["hyper_skinny_gemv_flag"], expected_hyper)
            if gemv_n96:
                self.assertEqual(sample["matmul_gemv_n96_flag"], {"control": "0", "candidate": "1"}[sample["mode"]])
            if subgroup_gate:
                self.assertEqual(sample["qmoe_subgroup_gate_flag"],
                                 {"control": "0", "candidate": "1"}.get(sample["mode"], configured_subgroup_gate))
                self.assertEqual(sample["matmul_gemv_n96_flag"], configured_gemv_n96)
            if bind_group_cache or dynamic_uniform_cache:
                self.assertEqual(sample["bind_group_cache_entries"],
                                 {"control": "0", "candidate": "8192"}.get(sample["mode"], configured_bind_group_cache))
                self.assertEqual(sample["matmul_gemv_n96_flag"], configured_gemv_n96)
                self.assertEqual(sample["qmoe_subgroup_gate_flag"], configured_subgroup_gate)
            expected_arena = ("0" if sample["mode"] == "control" else
                              "1" if sample["mode"] == "candidate" else arena) if uniform_arena else (
                                  arena if candidate else "0")
            self.assertEqual(sample["uniform_arena_flag"], expected_arena)
            self.assertEqual(sample["attention_subgroup_flag"], subgroup if candidate else "0")
            expected_split = ({"control": "0", "candidate": "1"}.get(sample["mode"], configured_split)
                              if attention_split else configured_split if candidate else "0")
            self.assertEqual(sample["attention_split_flag"], expected_split)
            expected_rank = ({"control": "0", "candidate": "1"}.get(sample["mode"], configured_rank)
                             if qsa_rank else configured_rank if candidate else "0")
            self.assertEqual(sample["qsa_parallel_rank_flag"], expected_rank)
        self.assertEqual(attention_flags, [(s["attention_subgroup_flag"], s["attention_split_flag"])
                                          for s in result["samples"]])
        self.assertEqual(hyper_skinny_values, [s["hyper_skinny_gemv_flag"] for s in result["samples"]])
        if gemv_n96:
            self.assertEqual(projection_flags, [(mode, configured_rank, configured_dense_map)
                                               for mode in ("1", "0", "0", "1", "1", "0")])
        if subgroup_gate:
            expected_gate_flags = ["1", "0", "0", "1", "1", "0"]
            if scoped:
                expected_gate_flags.extend([configured_subgroup_gate] * 3)
            self.assertEqual(subgroup_gate_flags, expected_gate_flags)
        if bind_group_cache or dynamic_uniform_cache:
            expected_cache_values = ["8192", "0", "0", "8192", "8192", "0"]
            if scoped:
                expected_cache_values.extend([configured_bind_group_cache] * 3)
            self.assertEqual(bind_group_cache_values, expected_cache_values)
        expected_overrides = ([] if validation_mode is None else
                              [("webgpu", "validationMode", validation_mode)])
        if max_pending is not None:
            expected_overrides.append(("webgpu", "maxNumPendingDispatches", str(max_pending)))
        self.assertEqual(provider_overrides, expected_overrides)
        self.assertEqual(result["webgpu_validation_mode_override"], validation_mode)
        self.assertEqual(result["webgpu_max_pending_dispatches_override"], max_pending)
        return result

    def test_dynamic_uniform_cache_bundle_changes_both_flags_and_preserves_other_experiments(self):
        self.invoke_comparison(
            cooperative="0", tail="1", upload="1", readback="1", qsa="1", arena="1",
            configured_rank="1", configured_dense_map="1", configured_gemv_n96="1",
            configured_split="1", dynamic_uniform_cache=True)

    def test_dynamic_uniform_cache_bundle_preserves_configured_noncomparison_context(self):
        self.invoke_comparison(
            cooperative="0", arena="1", scoped=True, dynamic_uniform_cache=True,
            configured_bind_group_cache="1", dynamic_uniforms="1")

    def test_hyper_skinny_comparison_preserves_other_flags_and_scoped_configuration(self):
        result = self.invoke_comparison(
            cooperative="0", tail="1", arena="1", configured_rank="1", configured_gemv_n96="1",
            configured_split="1", scoped=True, hyper_skinny=True, configured_hyper_skinny="1")
        self.assertEqual(result["comparison"], "hyper_skinny_only")

    def test_all_experiment_control_disables_hyper_skinny(self):
        self.invoke_comparison("0", configured_hyper_skinny="1")

    def test_webgpu_validation_override_is_applied_and_recorded(self):
        self.invoke_comparison("0", validation_mode="full")

    def test_pending_dispatch_override_is_applied_and_recorded(self):
        for value in (1, 64, 4096):
            with self.subTest(value=value):
                self.invoke_comparison("0", validation_mode="full", max_pending=value)

    def test_pending_dispatch_override_rejects_invalid_values_and_cuda(self):
        for extra in (["--webgpu-max-pending-dispatches", "0"],
                      ["--webgpu-max-pending-dispatches", "4097"],
                      ["--provider", "cuda", "--webgpu-max-pending-dispatches", "64"]):
            argv = ["benchmark", "--plugin", "candidate.so", "--output", "unused.json", *extra]
            with (self.subTest(extra=extra),
                  patch.dict(sys.modules, {"onnxruntime_genai": types.SimpleNamespace()}),
                  patch.object(sys, "argv", argv), patch("sys.stderr")):
                with self.assertRaises(SystemExit) as error:
                    benchmark.main()
                self.assertEqual(error.exception.code, 2)

    def test_webgpu_validation_override_rejects_cuda(self):
        argv = ["benchmark", "--plugin", "cuda.so", "--output", "unused.json",
                "--provider", "cuda", "--webgpu-validation-mode", "full"]
        with (patch.dict(sys.modules, {"onnxruntime_genai": types.SimpleNamespace()}),
              patch.object(sys, "argv", argv), patch("sys.stderr")):
            with self.assertRaises(SystemExit) as error:
                benchmark.main()
        self.assertEqual(error.exception.code, 2)

    def test_n96_comparison_varies_only_projection(self):
        result = self.invoke_comparison("0", tail="1", upload="1", readback="1", qsa="1", arena="1",
                                        configured_split="1", configured_rank="1", configured_dense_map="1",
                                        gemv_n96=True)
        self.assertEqual(result["comparison"], "gemv_n96_only")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_MATMUL_GEMV_N96"], "0")

    def test_subgroup_gate_comparison_preserves_fast_stack_and_scoped_configuration(self):
        result = self.invoke_comparison(
            "1", tail="1", upload="1", readback="1", qsa="1", arena="1",
            configured_split="1", configured_rank="1", configured_dense_map="1",
            subgroup_gate=True, configured_subgroup_gate="1", configured_gemv_n96="1", scoped=True)
        self.assertEqual(result["comparison"], "qmoe_subgroup_gate_only")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_QMOE_SUBGROUP_GATE"], "1")

    def test_binding_cache_comparison_preserves_other_experiments(self):
        result = self.invoke_comparison(
            "1", tail="1", upload="1", readback="1", qsa="1", arena="1",
            configured_split="1", configured_rank="1", configured_dense_map="1",
            configured_subgroup_gate="0", configured_gemv_n96="1", scoped=True,
            bind_group_cache=True, configured_bind_group_cache="8192")
        self.assertEqual(result["comparison"], "bind_group_cache_only")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES"], "8192")

    def test_dynamic_uniform_metadata_and_single_flag_comparison_preserve_configuration(self):
        result = self.invoke_comparison(
            "0", dynamic_uniforms="1", arena="1", bind_group_cache=True,
            configured_bind_group_cache="8192", scoped=True)
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_DYNAMIC_UNIFORMS"], "1")
        self.assertTrue(all(sample["dynamic_uniforms_flag"] == "1" for sample in result["samples"]))

    def test_all_experiment_control_disables_dynamic_uniforms(self):
        self.invoke_comparison("0", dynamic_uniforms="1")

    def test_qsa_rank_comparison_preserves_other_flags_and_scoped_configuration(self):
        result = self.invoke_comparison(
            "1", tail="1", upload="1", readback="1", qsa="1", configured_skinny="1", arena="1",
            qsa_rank=True, configured_rank="1", configured_split="1", subgroup="1", scoped=True)
        self.assertEqual(result["comparison"], "qsa_parallel_rank_only")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_QSA_PARALLEL_RANK"], "1")
        for sample in result["samples"]:
            self.assertEqual(sample["skinny_gemv_flag"], "1")
            self.assertEqual(sample["int4_prefill_flag"], "1")
            self.assertEqual(sample["int4_cooperative_flag"], "1")

    def test_all_experiment_comparison_disables_configured_rank_in_control(self):
        self.invoke_comparison("0", configured_rank="1")

    def test_attention_split_comparison_preserves_other_flags_and_scoped_configuration(self):
        result = self.invoke_comparison(
            "1", tail="1", upload="1", readback="1", qsa="1", configured_skinny="1", arena="1",
            attention_split=True, configured_split="1", subgroup="1", scoped=True)
        self.assertEqual(result["comparison"], "attention_split_only")
        for sample in result["samples"]:
            self.assertEqual(sample["skinny_gemv_flag"], "1")
            self.assertEqual(sample["int4_prefill_flag"], "1")
            self.assertEqual(sample["int4_cooperative_flag"], "1")
    def test_counterbalanced_flags_and_output_divergence(self):
        result = self.invoke_comparison("1")
        for sample in result["samples"]:
            self.assertFalse(sample["paired_output_matches"])
            self.assertEqual(sample["paired_first_different_token"], 1)

    def test_primary_comparison_keeps_cooperative_kernel_off(self):
        result = self.invoke_comparison("0")
        for sample in result["samples"]:
            self.assertTrue(sample["paired_output_matches"])
            self.assertIsNone(sample["paired_first_different_token"])

    def test_comparison_includes_dense_and_transport_flags(self):
        result = self.invoke_comparison("0", tail="1", upload="1")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_SMALL_UPLOAD_WRITE_BUFFER"], "1")
        self.assertTrue(all(sample["paired_output_matches"] for sample in result["samples"]))

    def test_expensive_context_runs_only_configured_candidate(self):
        result = self.invoke_comparison("0", tail="1", upload="1", scoped=True)
        samples = [sample for sample in result["samples"] if sample["context"] == 3]
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample["mode"] == "configured" for sample in samples))
        self.assertTrue(all("paired_output_matches" not in sample for sample in samples))

    def test_readback_and_qsa_flags_are_scoped_and_recorded(self):
        result = self.invoke_comparison("0", scoped=True, readback="1", qsa="1")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_SMALL_READBACK_CACHE"], "1")
        self.assertEqual(result["load_experiment_flags"]["ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE"], "1")

    def test_skinny_comparison_changes_only_skinny_flag(self):
        result = self.invoke_comparison("1", tail="1", upload="1", readback="1", qsa="1", skinny=True, arena="1")
        self.assertEqual(result["comparison"], "skinny_only")
        for sample in result["samples"]:
            self.assertEqual(sample["skinny_gemv_flag"], "1" if sample["mode"] == "candidate" else "0")
            self.assertEqual(sample["int4_prefill_flag"], "1")
            self.assertEqual(sample["int4_cooperative_flag"], "1")

    def test_uniform_arena_comparison_preserves_all_other_flags(self):
        result = self.invoke_comparison("1", tail="1", upload="1", readback="1", qsa="1",
                                        uniform_arena=True, configured_skinny="1", scoped=True)
        self.assertEqual(result["comparison"], "uniform_arena_only")
        for sample in result["samples"]:
            self.assertEqual(sample["skinny_gemv_flag"], "1")
            self.assertEqual(sample["int4_prefill_flag"], "1")
            self.assertEqual(sample["int4_cooperative_flag"], "1")


if __name__ == "__main__":
    unittest.main()
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
