from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_int4_candidate as candidate


class LaunchIntercepted(Exception):
    pass


class ReplayCandidateTests(unittest.TestCase):
    def test_packed_gate_windows_keep_clean_expert_parent_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate",
             "--mlp-input-projections", "--expert-bitcast-unpack", "--packed-gate"])[-1]
        self.assertIn("/qwen38-int4-qmoe-packed-gate/build-v3/libonnxruntime_providers_webgpu.so", shader_command)
        self.assertIn("--compare-packed-gate-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_PACKED_GATE=1", "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1",
                     "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1", "ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-hierarchical-gate-windows", shader_command)
        for flag in ("ORT_WEBGPU_DENSE_BITCAST_UNPACK=1", "ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS=1",
                     "ORT_WEBGPU_HC_UP_TILED=1"):
            self.assertNotIn(flag, shader_command)

    def test_packed_gate_rejects_unqualified_parent_combinations(self):
        for extra in ([], ["--expert-bitcast-unpack", "--dense-bitcast-unpack"],
                      ["--expert-bitcast-unpack", "--fc2-four-columns"],
                      ["--expert-bitcast-unpack", "--hc-up-tiled"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "profile",
                        "--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
                        "--post-norm-projection", "--hierarchical-gate", "--mlp-input-projections",
                        "--packed-gate", "--output-prefix", str(Path(temporary) / "candidate"), *extra]
                with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                      self.assertRaises(SystemExit) as error):
                    candidate.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()

    def test_hc_up_tiled_windows_keep_all_parent_fusions_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate",
             "--mlp-input-projections", "--expert-bitcast-unpack", "--hc-up-tiled"])[-1]
        self.assertIn("/qwen38-int4-hc-up-tiled/build-v1/libonnxruntime_providers_webgpu.so", shader_command)
        self.assertIn("--compare-hc-up-tiled-windows", shader_command)
        for flag in ("ORT_WEBGPU_HC_UP_TILED=1", "ORT_WEBGPU_HC_UP_PREMIX=1",
                     "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1", "ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-hc-up-premix-windows", shader_command)
        self.assertNotIn("ORT_WEBGPU_DENSE_BITCAST_UNPACK=1", shader_command)
        self.assertNotIn("ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS=1", shader_command)

    def test_hc_up_tiled_rejects_unqualified_parent_combinations(self):
        for extra in ([], ["--expert-bitcast-unpack", "--dense-bitcast-unpack"],
                      ["--expert-bitcast-unpack", "--fc2-four-columns"]):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as temporary:
                argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "profile",
                        "--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
                        "--post-norm-projection", "--hierarchical-gate", "--mlp-input-projections",
                        "--hc-up-tiled", "--output-prefix", str(Path(temporary) / "candidate"), *extra]
                with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                      self.assertRaises(SystemExit) as error):
                    candidate.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()

    def test_fc2_four_columns_windows_keep_clean_expert_parent_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate",
             "--mlp-input-projections", "--expert-bitcast-unpack", "--fc2-four-columns"])[-1]
        self.assertIn("/qwen38-int4-fc2-four-columns/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-fc2-four-columns-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_FC2_FOUR_COLUMNS=1", "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1",
                     "ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1", "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("ORT_WEBGPU_DENSE_BITCAST_UNPACK=1", shader_command)
        self.assertNotIn("--compare-expert-bitcast-unpack-windows", shader_command)

    def test_fc2_four_columns_requires_expert_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "profile",
                    "--fc2-four-columns", "--output-prefix", str(Path(temporary) / "candidate")]
            with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                  self.assertRaises(SystemExit) as error):
                candidate.main()
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_dense_unpack_windows_keep_expert_unpack_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate",
             "--mlp-input-projections", "--expert-bitcast-unpack", "--dense-bitcast-unpack"])[-1]
        self.assertIn("/qwen38-int4-dense-bitcast-unpack/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-dense-bitcast-unpack-windows", shader_command)
        for flag in ("ORT_WEBGPU_DENSE_BITCAST_UNPACK=1", "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1",
                     "ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1", "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-expert-bitcast-unpack-windows", shader_command)

    def test_dense_unpack_requires_expert_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "profile",
                    "--dense-bitcast-unpack", "--output-prefix", str(Path(temporary) / "candidate")]
            with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                  self.assertRaises(SystemExit) as error):
                candidate.main()
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_expert_unpack_windows_keep_mlp_and_all_fusions_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate",
             "--mlp-input-projections", "--expert-bitcast-unpack"])[-1]
        self.assertIn("/qwen38-int4-expert-bitcast-unpack/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-mlp-input-projections/model", shader_command)
        self.assertIn("--compare-expert-bitcast-unpack-windows", shader_command)
        for flag in ("ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1", "ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1",
                     "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_QSA_PARALLEL_MAP=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-mlp-bitcast-unpack-windows", shader_command)

    def test_bitcast_windows_toggle_only_unpacking_with_graph_fusion_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate",
             "--mlp-input-projections", "--mlp-bitcast-unpack"])[-1]
        self.assertIn("/qwen38-int4-mlp-bitcast-unpack/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-mlp-input-projections/model", shader_command)
        self.assertIn("--compare-mlp-bitcast-unpack-windows", shader_command)
        for flag in ("ORT_WEBGPU_MLP_BITCAST_UNPACK=1", "ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1",
                     "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1", "ORT_WEBGPU_QMOE_SUBGROUP_GATE=1",
                     "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_QSA_PARALLEL_MAP=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-hierarchical-gate-windows", shader_command)

    def test_bitcast_selector_requires_the_qualified_mlp_graph(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "profile",
                    "--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
                    "--post-norm-projection", "--hierarchical-gate", "--mlp-bitcast-unpack",
                    "--output-prefix", str(Path(temporary) / "candidate")]
            with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                  self.assertRaises(SystemExit) as error):
                candidate.main()
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_mlp_graph_fusion_preserves_parent_flags_and_selects_new_graph(self):
        shader_command = self.command("fc1-activation", "profile",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate", "--mlp-input-projections"])[-1]
        self.assertIn("/qwen38-int4-mlp-input-projections/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-mlp-input-projections/model", shader_command)
        for flag in ("ORT_WEBGPU_MLP_INPUT_PROJECTIONS=1", "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1",
                     "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_STATIC_NODE_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-", shader_command)

    def test_mlp_compound_fallback_is_not_used_for_performance_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "windows",
                    "--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
                    "--post-norm-projection", "--hierarchical-gate", "--mlp-input-projections",
                    "--output-prefix", str(Path(temporary) / "candidate")]
            with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                  self.assertRaises(SystemExit) as error):
                candidate.main()
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_hierarchical_windows_keep_subgroup_and_graph_fusions_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--hierarchical-gate"])[-1]
        self.assertIn("/qwen38-int4-qmoe-hierarchical-gate/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-post-norm-projection/model", shader_command)
        self.assertIn("--compare-hierarchical-gate-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1", "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1",
                     "ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_QSA_PARALLEL_MAP=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-subgroup-gate-windows", shader_command)

    def test_hierarchical_requires_unchanged_qualified_parent(self):
        for flags in ([], ["--post-norm-projection", "--post-norm-four-columns"]):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "profile",
                        "--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
                        "--hierarchical-gate", "--output-prefix", str(Path(temporary) / "candidate"), *flags]
                with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                      self.assertRaises(SystemExit) as error):
                    candidate.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()

    def test_postnorm_graph_variant_rejects_unfair_windows_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run_int4_candidate.py", "--kind", "fc1-activation", "--phase", "windows",
                    "--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
                    "--post-norm-projection", "--output-prefix", str(Path(temporary) / "candidate")]
            with (patch("sys.argv", argv), patch.object(candidate.subprocess, "run") as launch,
                  self.assertRaises(SystemExit) as error):
                candidate.main()
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_postnorm_graph_variant_uses_new_graph_and_keeps_parent_stack(self):
        shader_command = self.command("fc1-activation", "profile",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection"])[-1]
        self.assertIn("/qwen38-int4-hc-post-norm-projection/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-post-norm-projection/model", shader_command)
        for flag in ("ORT_WEBGPU_HC_POST_NORM_PROJECTION=1", "ORT_WEBGPU_QMOE_SUBGROUP_GATE=1",
                     "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1",
                     "ORT_WEBGPU_QSA_PARALLEL_MAP=1", "ORT_WEBGPU_HC_PROJECTION_SILU=1",
                     "ORT_WEBGPU_HC_UP_PREMIX=1", "ORT_WEBGPU_STATIC_NODE_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-", shader_command)

    def test_four_column_windows_keep_graph_fusion_and_parent_stack_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate",
             "--post-norm-projection", "--post-norm-four-columns"])[-1]
        self.assertIn("/qwen38-int4-hc-post-norm-projection/build-v3/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-post-norm-projection/model", shader_command)
        self.assertIn("--compare-hc-post-norm-four-columns-windows", shader_command)
        for flag in ("ORT_WEBGPU_HC_POST_NORM_FOUR_COLUMNS=1", "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1",
                     "ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_QSA_PARALLEL_MAP=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-subgroup-gate-windows", shader_command)

    def test_tiled_windows_expect_both_independent_replay_caches(self):
        self.assertEqual(candidate.projection_replay_counts(True, "windows"),
                         {"HCProjectionSiLUF16": 96, "HCProjectionSiLUTiledF16": 96})
        for phase in ("standard", "profile"):
            self.assertEqual(candidate.projection_replay_counts(True, phase),
                             {"HCProjectionSiLUTiledF16": 96})
        self.assertEqual(candidate.projection_replay_counts(False, "windows"),
                         {"HCProjectionSiLUF16": 96})

    def test_completed_guard_requires_exact_command_gpu_and_success(self):
        command = ["bash", "-lc", "expected"]
        guard = dict(status="completed", exit_code=0, command=command,
                     gpu_index=7, gpu_uuid=candidate.GPU_UUIDS[7])
        candidate.validate_completed_guard(guard, command, 7, candidate.GPU_UUIDS[7])
        for changed in (dict(status="timeout"), dict(exit_code=1),
                        dict(command=["bash", "-lc", "different"]),
                        dict(gpu_index=3), dict(gpu_uuid=candidate.GPU_UUIDS[3])):
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                candidate.validate_completed_guard(dict(guard, **changed), command, 7,
                                                   candidate.GPU_UUIDS[7])

    def command(self, kind, phase, extra=()):
        with tempfile.TemporaryDirectory() as temporary:
            argv = ["run_int4_candidate.py", "--kind", kind, "--phase", phase,
                    "--gpu-index", "3", "--output-prefix", str(Path(temporary) / "candidate")]
            argv.extend(extra)
            if kind == "hc-up-premix":
                argv.extend(["--up-revision", "v1"])
            with (patch("sys.argv", argv), patch.object(candidate, "snapshot", return_value={}),
                  patch.object(candidate, "prepare_model_cache", return_value={}),
                  patch.object(candidate.subprocess, "run", side_effect=LaunchIntercepted) as launch):
                with self.assertRaises(LaunchIntercepted):
                    candidate.main()
            self.assertEqual(launch.call_count, 1)
            return launch.call_args.args[0]

    def test_replay_uses_qualified_provider_original_graph_and_both_flags(self):
        command = self.command("node-replay", "profile")
        self.assertIn("--gpu-index", command)
        shader_command = command[-1]
        self.assertIn("/qwen38-int4-qmoe-node-replay/build-v5/libonnxruntime_providers_webgpu.so", shader_command)
        self.assertIn("--model /workspace/qwen38-int4-quickgelu-next/model", shader_command)
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", shader_command)
        self.assertIn("ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1", shader_command)
        self.assertIn("--profile-decode-after 31", shader_command)

    def test_combined_replay_checks_components_and_selects_fused_graph(self):
        shader_command = self.command("hc-node-replay", "profile")[-1]
        self.assertIn("/qwen38-int4-hc-node-replay-combined/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-post-norm/model", shader_command)
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", shader_command)

    def test_both_window_routes_toggle_replay_not_graph_or_gdn(self):
        for kind in ("node-replay", "hc-node-replay"):
            with self.subTest(kind=kind):
                shader_command = self.command(kind, "windows")[-1]
                self.assertIn("--tokens 288 --repetitions 4 --compare-node-replay-windows", shader_command)
                self.assertNotIn("--compare-gdn-view-windows", shader_command)
                self.assertIn("ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1", shader_command)

    def test_static_reuse_preserves_qmoe_and_toggles_only_static_programs(self):
        shader_command = self.command("static-replay", "windows")[-1]
        self.assertIn("/qwen38-int4-static-node-replay/build-v7/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-post-norm/model", shader_command)
        self.assertIn("--tokens 288 --repetitions 4 --compare-static-replay-windows", shader_command)
        self.assertIn("ORT_WEBGPU_STATIC_NODE_REPLAY=1", shader_command)
        self.assertIn("ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", shader_command)
        self.assertNotIn("--compare-node-replay-windows", shader_command)

    def test_projection_fusion_keeps_reuse_fixed_and_toggles_only_epilogue(self):
        shader_command = self.command("hc-projection", "windows")[-1]
        self.assertIn("/qwen38-int4-hc-projection-silu/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-projection-silu/model", shader_command)
        self.assertIn("--compare-hc-projection-windows", shader_command)
        for flag in ("ORT_WEBGPU_STATIC_NODE_REPLAY=1", "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1",
                     "ORT_WEBGPU_HC_PROJECTION_SILU=1", "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-static-replay-windows", shader_command)

    def test_up_premix_keeps_previous_projection_fusion_fixed(self):
        shader_command = self.command("hc-up-premix", "windows")[-1]
        self.assertIn("/qwen38-int4-hc-up-premix/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-hc-up-premix-windows", shader_command)
        self.assertIn("ORT_WEBGPU_HC_PROJECTION_SILU=1", shader_command)
        self.assertIn("ORT_WEBGPU_HC_UP_PREMIX=1", shader_command)
        self.assertNotIn("--compare-hc-projection-windows", shader_command)

    def test_parallel_map_uses_qualified_parent_and_only_toggles_map(self):
        shader_command = self.command("qsa-map", "windows")[-1]
        self.assertIn("/qwen38-int4-qsa-parallel-map/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-qsa-parallel-map-windows", shader_command)
        for flag in ("ORT_WEBGPU_QSA_PARALLEL_MAP=1", "ORT_WEBGPU_HC_PROJECTION_SILU=1",
                     "ORT_WEBGPU_HC_UP_PREMIX=1", "ORT_WEBGPU_STATIC_NODE_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-hc-up-premix-windows", shader_command)

    def test_generic_replay_uses_parent_fusions_and_only_toggles_generic_nodes(self):
        shader_command = self.command("generic-replay", "windows")[-1]
        self.assertIn("/qwen38-int4-generic-matmul-replay/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-generic-matmul-windows", shader_command)
        for flag in ("ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1", "ORT_WEBGPU_HC_PROJECTION_SILU=1",
                     "ORT_WEBGPU_HC_UP_PREMIX=1", "ORT_WEBGPU_STATIC_NODE_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-hc-up-premix-windows", shader_command)

    def test_direct_a_preserves_replay_and_parent_fusions(self):
        shader_command = self.command("direct-a", "windows")[-1]
        self.assertIn("/qwen38-int4-direct-a-matmul/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-direct-a-windows", shader_command)
        for flag in ("ORT_WEBGPU_MATMUL_NBITS_DIRECT_A=1", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1",
                     "ORT_WEBGPU_HC_PROJECTION_SILU=1", "ORT_WEBGPU_HC_UP_PREMIX=1",
                     "ORT_WEBGPU_STATIC_NODE_REPLAY=1", "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-generic-matmul-windows", shader_command)

    def test_fc1_activation_preserves_parent_and_toggles_only_epilogue(self):
        shader_command = self.command("fc1-activation", "windows")[-1]
        self.assertIn("/qwen38-int4-qmoe-fc1-activation/build-v3/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-fc1-activation-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1",
                     "ORT_WEBGPU_HC_PROJECTION_SILU=1", "ORT_WEBGPU_HC_UP_PREMIX=1",
                     "ORT_WEBGPU_STATIC_NODE_REPLAY=1", "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1",
                     "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-generic-matmul-windows", shader_command)
        self.assertNotIn("ORT_WEBGPU_MATMUL_NBITS_DIRECT_A=", shader_command)
        self.assertNotIn("ORT_WEBGPU_QSA_PARALLEL_MAP=", shader_command)

    def test_immediate_encoding_keeps_fc1_and_all_parent_fusions_enabled(self):
        shader_command = self.command("fc1-activation", "windows", ["--replay-immediate"])[-1]
        self.assertIn("/qwen38-int4-node-replay-immediate/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-replay-immediate-windows", shader_command)
        for flag in ("ORT_WEBGPU_NODE_REPLAY_IMMEDIATE=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1", "ORT_WEBGPU_HC_PROJECTION_SILU=1",
                     "ORT_WEBGPU_HC_UP_PREMIX=1", "ORT_WEBGPU_STATIC_NODE_REPLAY=1",
                     "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-fc1-activation-windows", shader_command)
        self.assertNotIn("ORT_WEBGPU_QSA_PARALLEL_MAP=", shader_command)

    def test_fc2_final_mix_keeps_fc1_enabled_without_immediate_encoding(self):
        shader_command = self.command("fc1-activation", "windows", ["--fc2-final-mix"])[-1]
        self.assertIn("/qwen38-int4-qmoe-fc2-final-mix/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-fc2-final-mix-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1", "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1",
                     "ORT_WEBGPU_STATIC_NODE_REPLAY=1", "ORT_WEBGPU_HC_PROJECTION_SILU=1",
                     "ORT_WEBGPU_HC_UP_PREMIX=1", "ORT_WEBGPU_GDN_PACKED_DECODE_VIEW=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-fc1-activation-windows", shader_command)
        self.assertNotIn("ORT_WEBGPU_NODE_REPLAY_IMMEDIATE=", shader_command)
        self.assertNotIn("ORT_WEBGPU_QSA_PARALLEL_MAP=", shader_command)

    def test_combined_map_windows_keep_both_expert_fusions_enabled(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--context", "8192"])[-1]
        self.assertIn("/qwen38-int4-qmoe-fusions-map/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--model /workspace/qwen38-int4-hc-up-premix/model", shader_command)
        self.assertIn("--compare-qsa-parallel-map-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1",
                     "ORT_WEBGPU_QSA_PARALLEL_MAP=1", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1",
                     "ORT_WEBGPU_HC_PROJECTION_SILU=1", "ORT_WEBGPU_HC_UP_PREMIX=1",
                     "ORT_WEBGPU_STATIC_NODE_REPLAY=1", "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-fc2-final-mix-windows", shader_command)
        self.assertNotIn("ORT_WEBGPU_NODE_REPLAY_IMMEDIATE=", shader_command)

    def test_padding_windows_toggle_layout_without_disabling_fusions(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--padded-partials"])[-1]
        self.assertIn("/qwen38-int4-qmoe-fc2-padding/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-fc2-padded-partials-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1",
                     "ORT_WEBGPU_QMOE_FC2_PADDED_PARTIALS=1", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-fc2-final-mix-windows", shader_command)
        self.assertNotIn("ORT_WEBGPU_QSA_PARALLEL_MAP=", shader_command)

    def test_tiled_projection_keeps_the_qualified_combined_stack_enabled(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--tiled-projection"])[-1]
        self.assertIn("/qwen38-int4-hc-projection-tiled/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-hc-projection-tiled-windows", shader_command)
        for flag in ("ORT_WEBGPU_HC_PROJECTION_TILED=1", "ORT_WEBGPU_HC_PROJECTION_SILU=1",
                     "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1",
                     "ORT_WEBGPU_QSA_PARALLEL_MAP=1", "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-qsa-parallel-map-windows", shader_command)
        self.assertNotIn("--compare-fc2-final-mix-windows", shader_command)

    def test_subgroup_gate_windows_keep_both_expert_fusions_and_map_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate"])[-1]
        self.assertIn("/qwen38-int4-qmoe-fusions-map/build-v2/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-subgroup-gate-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_SUBGROUP_GATE=1", "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1",
                     "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1", "ORT_WEBGPU_QSA_PARALLEL_MAP=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-qsa-parallel-map-windows", shader_command)
        self.assertNotIn("--compare-fc2-final-mix-windows", shader_command)

    def test_activation_padding_keeps_subgroup_gate_and_combined_stack_fixed(self):
        shader_command = self.command("fc1-activation", "windows",
            ["--fc2-final-mix", "--with-parallel-map", "--subgroup-gate", "--padded-activations"])[-1]
        self.assertIn("/qwen38-int4-qmoe-activation-padding/build-v1/libonnxruntime_providers_webgpu.so",
                      shader_command)
        self.assertIn("--compare-qmoe-activation-padding-windows", shader_command)
        for flag in ("ORT_WEBGPU_QMOE_PADDED_ACTIVATIONS=1", "ORT_WEBGPU_QMOE_SUBGROUP_GATE=1",
                     "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1",
                     "ORT_WEBGPU_QSA_PARALLEL_MAP=1"):
            self.assertIn(flag, shader_command)
        self.assertNotIn("--compare-subgroup-gate-windows", shader_command)
        self.assertNotIn("--compare-qsa-parallel-map-windows", shader_command)


if __name__ == "__main__":
    unittest.main()
