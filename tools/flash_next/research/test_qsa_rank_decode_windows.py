import copy
import os
import types
import unittest
from unittest.mock import patch

import qsa_rank_decode_windows as windows


class RankWindowTests(unittest.TestCase):
    def test_dense_qkv_pad34_windows_toggle_and_restore(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_QKV_PAD34_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_QKV_PAD34_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_qkv_pad340_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_qkv_pad341_median_window_tps"], 2.0)

    def test_dense_qkv_pad34_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_QKV_PAD34_FLAG, interrupt=True)

    def test_dense_qkv_threads64_windows_toggle_and_restore(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_QKV_THREADS64_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_QKV_THREADS64_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_qkv_threads640_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_qkv_threads641_median_window_tps"], 2.0)

    def test_dense_qkv_threads64_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_QKV_THREADS64_FLAG, interrupt=True)

    def test_dense_qkv_predecode8_windows_toggle_and_restore(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_QKV_PREDECODE8_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_QKV_PREDECODE8_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_qkv_predecode80_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_qkv_predecode81_median_window_tps"], 2.0)

    def test_dense_qkv_predecode8_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_QKV_PREDECODE8_FLAG, interrupt=True)

    def test_dense_qkv_k16_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_QKV_K16_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_QKV_K16_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_qkv_k160_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_qkv_k161_median_window_tps"], 2.0)

    def test_dense_qkv_k16_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_QKV_K16_FLAG, interrupt=True)

    def test_paired_and_half_tile_fc1_windows_toggle_and_restore(self):
        for flag, prefix in ((windows.FC1_PAIRED_DOT_FLAG, "fc1_paired_dot"),
                             (windows.FC1_K16_TILE_FLAG, "fc1_k16_tile"),
                             (windows.FC1_K8_TILE_FLAG, "fc1_k8_tile"),
                             (windows.FC1_U32_PARTIALS_FLAG, "fc1_u32_partials"),
                             (windows.FC1_EXPERT_PAIRS_FLAG, "fc1_expert_pairs")):
            with self.subTest(flag=flag):
                result, _ = self.invoke(experiment_flag=flag)
                self.assertEqual(result["window_protocol"]["experiment_flag"], flag)
                summary = result["window_summary"]["32768"]
                self.assertEqual(summary[f"{prefix}0_median_window_tps"], 1.0)
                self.assertEqual(summary[f"{prefix}1_median_window_tps"], 2.0)

    def test_paired_and_half_tile_fc1_interruption_restores_environment(self):
        for flag in (windows.FC1_PAIRED_DOT_FLAG, windows.FC1_K16_TILE_FLAG, windows.FC1_K8_TILE_FLAG,
                     windows.FC1_U32_PARTIALS_FLAG, windows.FC1_EXPERT_PAIRS_FLAG):
            with self.subTest(flag=flag), self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                self.invoke(experiment_flag=flag, interrupt=True)

    def test_fc1_columns8_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC1_COLUMNS8_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC1_COLUMNS8_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["fc1_columns80_median_window_tps"], 1.0)
        self.assertEqual(summary["fc1_columns81_median_window_tps"], 2.0)

    def test_fc1_columns8_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC1_COLUMNS8_FLAG, interrupt=True)

    def test_fc1_full_a_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC1_FULL_A_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC1_FULL_A_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["fc1_full_a0_median_window_tps"], 1.0)
        self.assertEqual(summary["fc1_full_a1_median_window_tps"], 2.0)

    def test_fc1_full_a_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC1_FULL_A_FLAG, interrupt=True)

    def test_fc1_experts10_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC1_EXPERTS10_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC1_EXPERTS10_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["fc1_experts100_median_window_tps"], 1.0)
        self.assertEqual(summary["fc1_experts101_median_window_tps"], 2.0)

    def test_fc1_experts10_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC1_EXPERTS10_FLAG, interrupt=True)

    def test_dense_qkv_wide16_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_QKV_WIDE16_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_QKV_WIDE16_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_qkv_wide160_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_qkv_wide161_median_window_tps"], 2.0)

    def test_dense_qkv_wide16_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_QKV_WIDE16_FLAG, interrupt=True)

    def test_fc2_compact20_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC2_COMPACT20_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC2_COMPACT20_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["fc2_compact200_median_window_tps"], 1.0)
        self.assertEqual(summary["fc2_compact201_median_window_tps"], 2.0)

    def test_fc2_compact20_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC2_COMPACT20_FLAG, interrupt=True)

    def test_fc2_subgroup_fold_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC2_SUBGROUP_FOLD_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC2_SUBGROUP_FOLD_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["fc2_subgroup_fold0_median_window_tps"], 1.0)
        self.assertEqual(summary["fc2_subgroup_fold1_median_window_tps"], 2.0)

    def test_fc2_subgroup_fold_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC2_SUBGROUP_FOLD_FLAG, interrupt=True)

    def test_dense_predecode_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_PREDECODE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_PREDECODE_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_predecode0_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_predecode1_median_window_tps"], 2.0)

    def test_dense_predecode_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_PREDECODE_FLAG, interrupt=True)

    def test_gdn_norm32_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.GDN_NORM32_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.GDN_NORM32_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["gdn_norm320_median_window_tps"], 1.0)
        self.assertEqual(summary["gdn_norm321_median_window_tps"], 2.0)

    def test_gdn_norm32_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.GDN_NORM32_FLAG, interrupt=True)

    def test_gdn_tree32_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.GDN_TREE32_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.GDN_TREE32_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["gdn_tree320_median_window_tps"], 1.0)
        self.assertEqual(summary["gdn_tree321_median_window_tps"], 2.0)

    def test_gdn_tree32_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.GDN_TREE32_FLAG, interrupt=True)

    def test_hc_post_norm_predecode_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_POST_NORM_PREDECODE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_POST_NORM_PREDECODE_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_post_norm_predecode0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_post_norm_predecode1_median_window_tps"], 2.0)

    def test_hc_post_norm_predecode_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_POST_NORM_PREDECODE_FLAG, interrupt=True)

    def test_gdn_value8_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.GDN_VALUE8_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.GDN_VALUE8_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["gdn_value80_median_window_tps"], 1.0)
        self.assertEqual(summary["gdn_value81_median_window_tps"], 2.0)

    def test_gdn_value8_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.GDN_VALUE8_FLAG, interrupt=True)

    def test_hc_post_norm_vec4_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_POST_NORM_VEC4_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_POST_NORM_VEC4_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_post_norm_vec40_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_post_norm_vec41_median_window_tps"], 2.0)

    def test_hc_post_norm_vec4_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_POST_NORM_VEC4_FLAG, interrupt=True)

    def test_dsa_paired_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DSA_PAIRED_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DSA_PAIRED_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dsa_paired0_median_window_tps"], 1.0)
        self.assertEqual(summary["dsa_paired1_median_window_tps"], 2.0)

    def test_dsa_paired_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DSA_PAIRED_FLAG, interrupt=True)

    def test_dense_block32_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_BLOCK32_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_BLOCK32_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_block320_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_block321_median_window_tps"], 2.0)

    def test_dense_block32_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_BLOCK32_FLAG, interrupt=True)

    def test_dense_static_shape_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_STATIC_SHAPE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_STATIC_SHAPE_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_static_shape0_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_static_shape1_median_window_tps"], 2.0)

    def test_dense_static_shape_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_STATIC_SHAPE_FLAG, interrupt=True)

    def test_qsa_pruned_rank_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QSA_PRUNED_RANK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QSA_PRUNED_RANK_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qsa_pruned_rank0_median_window_tps"], 1.0)
        self.assertEqual(summary["qsa_pruned_rank1_median_window_tps"], 2.0)

    def test_qsa_pruned_rank_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QSA_PRUNED_RANK_FLAG, interrupt=True)

    def test_qsa_subgroup_rank_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QSA_SUBGROUP_RANK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QSA_SUBGROUP_RANK_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qsa_subgroup_rank0_median_window_tps"], 1.0)
        self.assertEqual(summary["qsa_subgroup_rank1_median_window_tps"], 2.0)

    def test_qsa_subgroup_rank_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QSA_SUBGROUP_RANK_FLAG, interrupt=True)

    def test_gdn_value2_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.GDN_VALUE2_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.GDN_VALUE2_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["gdn_value20_median_window_tps"], 1.0)
        self.assertEqual(summary["gdn_value21_median_window_tps"], 2.0)

    def test_gdn_value2_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.GDN_VALUE2_FLAG, interrupt=True)

    def test_dense_double_buffer_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_DOUBLE_BUFFER_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_DOUBLE_BUFFER_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_double_buffer0_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_double_buffer1_median_window_tps"], 2.0)

    def test_dense_double_buffer_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_DOUBLE_BUFFER_FLAG, interrupt=True)

    def test_dsa_query_cache_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DSA_QUERY_CACHE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DSA_QUERY_CACHE_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dsa_query_cache0_median_window_tps"], 1.0)
        self.assertEqual(summary["dsa_query_cache1_median_window_tps"], 2.0)

    def test_dsa_query_cache_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DSA_QUERY_CACHE_FLAG, interrupt=True)

    def test_dsa_partition32_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DSA_PARTITION32_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DSA_PARTITION32_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dsa_partition320_median_window_tps"], 1.0)
        self.assertEqual(summary["dsa_partition321_median_window_tps"], 2.0)

    def test_dsa_partition32_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DSA_PARTITION32_FLAG, interrupt=True)

    def test_qmoe_fast_flags_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QMOE_FAST_FLAGS_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QMOE_FAST_FLAGS_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qmoe_fast_flags0_median_window_tps"], 1.0)
        self.assertEqual(summary["qmoe_fast_flags1_median_window_tps"], 2.0)

    def test_qmoe_fast_flags_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QMOE_FAST_FLAGS_FLAG, interrupt=True)

    def test_dense_preload_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_PRELOAD_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_PRELOAD_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_preload0_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_preload1_median_window_tps"], 2.0)

    def test_dense_preload_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_PRELOAD_FLAG, interrupt=True)

    def test_hc_parallel4_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_UP_PARALLEL4_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_UP_PARALLEL4_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_up_parallel40_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_up_parallel41_median_window_tps"], 2.0)

    def test_hc_parallel4_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_UP_PARALLEL4_FLAG, interrupt=True)

    def test_hc_staged_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_STAGED_PROJECTION_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_STAGED_PROJECTION_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_staged_projection0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_staged_projection1_median_window_tps"], 2.0)

    def test_hc_staged_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_STAGED_PROJECTION_FLAG, interrupt=True)

    def test_dense_wide8_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_WIDE8_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_WIDE8_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_wide80_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_wide81_median_window_tps"], 2.0)

    def test_dense_wide8_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_WIDE8_FLAG, interrupt=True)

    def test_dense_tile2_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_TILE2_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_TILE2_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_tile20_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_tile21_median_window_tps"], 2.0)

    def test_dense_tile2_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_TILE2_FLAG, interrupt=True)

    def test_projection_shuffle_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_PROJECTION_SHUFFLE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_PROJECTION_SHUFFLE_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_projection_shuffle0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_projection_shuffle1_median_window_tps"], 2.0)

    def test_projection_shuffle_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_PROJECTION_SHUFFLE_FLAG, interrupt=True)

    def test_dense_tile4_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_TILE4_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_TILE4_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_tile40_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_tile41_median_window_tps"], 2.0)

    def test_dense_tile4_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DENSE_TILE4_FLAG, interrupt=True)

    def test_postnorm_padding_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_POST_NORM_PAD34_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_POST_NORM_PAD34_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_post_norm_pad340_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_post_norm_pad341_median_window_tps"], 2.0)

    def test_postnorm_padding_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_POST_NORM_PAD34_FLAG, interrupt=True)

    def test_hc_up_parallel2_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_UP_PARALLEL2_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_UP_PARALLEL2_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_up_parallel20_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_up_parallel21_median_window_tps"], 2.0)

    def test_hc_up_parallel2_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_UP_PARALLEL2_FLAG, interrupt=True)

    def test_score16_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QSA_SCORE16_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QSA_SCORE16_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qsa_score160_median_window_tps"], 1.0)
        self.assertEqual(summary["qsa_score161_median_window_tps"], 2.0)

    def test_score16_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QSA_SCORE16_FLAG, interrupt=True)

    def test_indexer_stack_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QSA_INDEXER_STACK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QSA_INDEXER_STACK_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qsa_indexer_stack0_median_window_tps"], 1.0)
        self.assertEqual(summary["qsa_indexer_stack1_median_window_tps"], 2.0)

    def test_indexer_stack_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QSA_INDEXER_STACK_FLAG, interrupt=True)

    def test_packed_score_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QSA_PACKED_SCORE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QSA_PACKED_SCORE_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qsa_packed_score0_median_window_tps"], 1.0)
        self.assertEqual(summary["qsa_packed_score1_median_window_tps"], 2.0)

    def test_packed_score_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QSA_PACKED_SCORE_FLAG, interrupt=True)

    def test_concat_vector_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.QSA_CONCAT_VECTOR_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.QSA_CONCAT_VECTOR_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["qsa_concat_vector0_median_window_tps"], 1.0)
        self.assertEqual(summary["qsa_concat_vector1_median_window_tps"], 2.0)

    def test_concat_vector_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.QSA_CONCAT_VECTOR_FLAG, interrupt=True)

    def test_hc_up_bitcast_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_UP_BITCAST_UNPACK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_UP_BITCAST_UNPACK_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_up_bitcast_unpack0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_up_bitcast_unpack1_median_window_tps"], 2.0)

    def test_hc_up_bitcast_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_UP_BITCAST_UNPACK_FLAG, interrupt=True)

    def test_hc_subgroup_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_POST_NORM_SUBGROUP_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_POST_NORM_SUBGROUP_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_post_norm_subgroup0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_post_norm_subgroup1_median_window_tps"], 2.0)

    def test_hc_subgroup_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_POST_NORM_SUBGROUP_FLAG, interrupt=True)

    def test_hc_fused_sum_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_POST_NORM_FUSED_SUM_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_POST_NORM_FUSED_SUM_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_post_norm_fused_sum0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_post_norm_fused_sum1_median_window_tps"], 2.0)

    def test_hc_fused_sum_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_POST_NORM_FUSED_SUM_FLAG, interrupt=True)

    def test_hc_bitcast_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_POST_NORM_BITCAST_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_POST_NORM_BITCAST_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_post_norm_bitcast0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_post_norm_bitcast1_median_window_tps"], 2.0)

    def test_hc_bitcast_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_POST_NORM_BITCAST_FLAG, interrupt=True)

    def test_packed_gate_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.PACKED_GATE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.PACKED_GATE_FLAG)
        self.assertEqual({row["packed_gate_flag"] for row in result["window_samples"]}, {0, 1})
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["packed_gate0_median_window_tps"], 1.0)
        self.assertEqual(summary["packed_gate1_median_window_tps"], 2.0)

    def test_packed_gate_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.PACKED_GATE_FLAG, interrupt=True)

    def test_hc_up_tiled_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_UP_TILED_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_UP_TILED_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hc_up_tiled0_median_window_tps"], 1.0)
        self.assertEqual(summary["hc_up_tiled1_median_window_tps"], 2.0)

    def test_fc2_four_columns_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC2_FOUR_COLUMNS_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC2_FOUR_COLUMNS_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["fc2_four_columns0_median_window_tps"], 1.0)
        self.assertEqual(summary["fc2_four_columns1_median_window_tps"], 2.0)

    def test_dense_bitcast_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_BITCAST_UNPACK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_BITCAST_UNPACK_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_bitcast_unpack0_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_bitcast_unpack1_median_window_tps"], 2.0)

    def test_expert_bitcast_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.EXPERT_BITCAST_UNPACK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.EXPERT_BITCAST_UNPACK_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["geometric_mean_matched_window_ratio"], 2.0)

    def test_bitcast_unpack_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.MLP_BITCAST_UNPACK_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.MLP_BITCAST_UNPACK_FLAG)
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["mlp_bitcast_unpack0_median_window_tps"], 1.0)
        self.assertEqual(summary["mlp_bitcast_unpack1_median_window_tps"], 2.0)

    def test_hierarchical_gate_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HIERARCHICAL_GATE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HIERARCHICAL_GATE_FLAG)
        self.assertEqual({row["hierarchical_gate_flag"] for row in result["window_samples"]}, {0, 1})
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["hierarchical_gate0_median_window_tps"], 1.0)
        self.assertEqual(summary["hierarchical_gate1_median_window_tps"], 2.0)

    def test_hierarchical_gate_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HIERARCHICAL_GATE_FLAG, interrupt=True)

    def invoke(self, *, mismatch=False, interrupt=False, second_request_slowdown=1,
               experiment_flag=windows.RANK_FLAG):
        args = types.SimpleNamespace(tokens=288, contexts=[32768], repetitions=2, chunk=16)
        result = {"samples": []}
        snapshots = []
        clock = 0.0
        calls = 0
        prompts = []

        def request(og, model, tokenizer, prompt, output_tokens, chunk, *, on_decode_step):
            nonlocal clock, calls
            calls += 1
            prompts.append(list(prompt))
            for index in range(output_tokens):
                if experiment_flag == windows.DENSE_MAP_FLAG:
                    self.assertEqual(os.environ[windows.RANK_FLAG], "1")
                if experiment_flag in (windows.EXACT_MID_FLAG, windows.EXACT_WIDE_FLAG):
                    self.assertEqual(os.environ["ORT_WEBGPU_HC_EXACT_DENSE"], "1")
                    self.assertEqual(os.environ[windows.RANK_FLAG], "1")
                    self.assertEqual(os.environ[windows.DENSE_MAP_FLAG], "1")
                if experiment_flag == windows.EXACT_WIDE_FLAG:
                    self.assertEqual(os.environ[windows.EXACT_MID_FLAG], "1")
                step = 1.0 if os.environ[experiment_flag] == "0" else 0.5
                clock += step * (second_request_slowdown if calls == 2 else 1)
                on_decode_step(index, clock)
                if interrupt and index == 64:
                    raise RuntimeError("simulated interrupted request")
            tokens = list(range(output_tokens))
            if mismatch and calls == 2:
                tokens[-1] = -1
            return {"input_tokens": len(prompt), "output_tokens": output_tokens, "output_token_ids": tokens}

        with (
            patch.dict(os.environ, {windows.RANK_FLAG: "1", windows.DENSE_MAP_FLAG: "1",
                                    windows.EXACT_MID_FLAG: "1",
                                    "ORT_WEBGPU_HC_EXACT_DENSE": "1", experiment_flag: "previous"}),
            patch.object(windows, "natural_text_prompt", return_value=([10, 20], 7)),
            patch.object(windows, "run_request", side_effect=request),
            patch.object(windows.time, "perf_counter", side_effect=lambda: clock),
        ):
            try:
                windows.run_qsa_windows(None, None, None, [], args, result,
                                        lambda: snapshots.append(copy.deepcopy(result)),
                                        experiment_flag=experiment_flag)
            finally:
                self.assertEqual(os.environ[experiment_flag], "previous")
                self.last_result = result
                self.last_snapshots = snapshots
        self.assertEqual(prompts, [[10, 20], [10, 20]])
        return result, snapshots

    def test_matching_context_windows_are_balanced_and_measured_after_warmup(self):
        result, snapshots = self.invoke()
        self.assertEqual(len(result["window_samples"]), 16)
        self.assertEqual([row["rank_flag"] for row in result["window_samples"][:8]],
                         list(windows.WINDOW_PATTERN))
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["rank0_median_window_tps"], 1.0)
        self.assertEqual(summary["rank1_median_window_tps"], 2.0)
        self.assertEqual(summary["geometric_mean_matched_window_ratio"], 2.0)
        self.assertEqual(summary["matched_window_pairs"], 8)
        self.assertEqual(result["window_samples"][0]["logical_sequence_tokens_before"], 32800)
        self.assertTrue(all(row["same_prompt_output_matches"] for row in result["samples"]))
        first_completed = next(s for s in snapshots if len(s["samples"]) == 1)
        self.assertIsNone(first_completed["samples"][0]["same_prompt_output_matches"])

    def test_different_generated_tokens_fail_without_publishing_a_summary(self):
        with self.assertRaisesRegex(RuntimeError, "generated different tokens"):
            self.invoke(mismatch=True)
        self.assertFalse(self.last_result["samples"][-1]["same_prompt_output_matches"])
        self.assertNotIn("window_summary", self.last_result)

    def test_multiplicative_request_slowdown_is_visible_and_does_not_inflate_paired_ratio(self):
        result, _ = self.invoke(second_request_slowdown=100)
        summary = result["window_summary"]["32768"]
        self.assertAlmostEqual(summary["geometric_mean_matched_window_ratio"], 2.0)
        self.assertEqual(summary["per_request"][0]["rank0_median_window_tps"], 1.0)
        self.assertEqual(summary["per_request"][1]["rank0_median_window_tps"], 0.01)

    def test_interrupted_request_retains_only_completed_windows(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(interrupt=True)
        self.assertEqual(len(self.last_result["window_samples"]), 1)
        self.assertEqual(self.last_result["samples"], [])
        self.assertNotIn("window_summary", self.last_result)
        self.assertTrue(any(len(s["window_samples"]) == 1 for s in self.last_snapshots))

    def test_dense_map_windows_keep_rank_fixed_and_label_the_actual_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DENSE_MAP_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DENSE_MAP_FLAG)
        self.assertTrue(all("dense_map_flag" in row and "rank_flag" not in row
                            for row in result["window_samples"]))
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["dense_map0_median_window_tps"], 1.0)
        self.assertEqual(summary["dense_map1_median_window_tps"], 2.0)
        self.assertEqual(summary["geometric_mean_matched_window_ratio"], 2.0)

    def test_unsupported_experiment_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported decode-window"):
            windows.run_qsa_windows(None, None, None, [], None, {}, lambda: None,
                                    experiment_flag="UNSUPPORTED")

    def test_gdn_view_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.GDN_VIEW_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.GDN_VIEW_FLAG)
        self.assertEqual({row["gdn_view_flag"] for row in result["window_samples"]}, {0, 1})
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["gdn_view0_median_window_tps"], 1.0)
        self.assertEqual(summary["gdn_view1_median_window_tps"], 2.0)

    def test_node_replay_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.NODE_REPLAY_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.NODE_REPLAY_FLAG)
        self.assertEqual({row["node_replay_flag"] for row in result["window_samples"]}, {0, 1})
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["node_replay0_median_window_tps"], 1.0)
        self.assertEqual(summary["node_replay1_median_window_tps"], 2.0)

    def test_node_replay_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.NODE_REPLAY_FLAG, interrupt=True)

    def test_static_replay_windows_toggle_and_restore_exact_flag(self):
        result, _ = self.invoke(experiment_flag=windows.STATIC_REPLAY_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.STATIC_REPLAY_FLAG)
        self.assertEqual({row["static_replay_flag"] for row in result["window_samples"]}, {0, 1})
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["static_replay0_median_window_tps"], 1.0)
        self.assertEqual(summary["static_replay1_median_window_tps"], 2.0)

    def test_static_replay_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.STATIC_REPLAY_FLAG, interrupt=True)

    def test_hc_projection_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_PROJECTION_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_PROJECTION_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["hc_projection1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_PROJECTION_FLAG, interrupt=True)

    def test_up_premix_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_UP_PREMIX_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_UP_PREMIX_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["hc_up_premix1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_UP_PREMIX_FLAG, interrupt=True)

    def test_parallel_map_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.PARALLEL_MAP_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.PARALLEL_MAP_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["parallel_map1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.PARALLEL_MAP_FLAG, interrupt=True)

    def test_generic_matmul_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.GENERIC_MATMUL_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.GENERIC_MATMUL_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["generic_matmul1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.GENERIC_MATMUL_FLAG, interrupt=True)

    def test_direct_a_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.DIRECT_A_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.DIRECT_A_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["direct_a1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.DIRECT_A_FLAG, interrupt=True)

    def test_fc1_activation_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC1_ACTIVATION_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC1_ACTIVATION_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["fc1_activation1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC1_ACTIVATION_FLAG, interrupt=True)

    def test_immediate_replay_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.REPLAY_IMMEDIATE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.REPLAY_IMMEDIATE_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["replay_immediate1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.REPLAY_IMMEDIATE_FLAG, interrupt=True)

    def test_fc2_final_mix_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC2_FINAL_MIX_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC2_FINAL_MIX_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["fc2_final_mix1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC2_FINAL_MIX_FLAG, interrupt=True)

    def test_fc2_padding_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.FC2_PADDED_PARTIALS_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.FC2_PADDED_PARTIALS_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["fc2_padded_partials1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.FC2_PADDED_PARTIALS_FLAG, interrupt=True)

    def test_tiled_projection_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.HC_PROJECTION_TILED_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.HC_PROJECTION_TILED_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["hc_projection_tiled1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.HC_PROJECTION_TILED_FLAG, interrupt=True)

    def test_subgroup_gate_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.SUBGROUP_GATE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.SUBGROUP_GATE_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["subgroup_gate1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.SUBGROUP_GATE_FLAG, interrupt=True)

    def test_activation_padding_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.ACTIVATION_PADDING_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.ACTIVATION_PADDING_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["activation_padding1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.ACTIVATION_PADDING_FLAG, interrupt=True)

    def test_postnorm_four_columns_windows_and_interruption_restore_flag(self):
        result, _ = self.invoke(experiment_flag=windows.POST_NORM_FOUR_COLUMNS_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.POST_NORM_FOUR_COLUMNS_FLAG)
        self.assertEqual(result["window_summary"]["32768"]["post_norm_four_columns1_median_window_tps"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.POST_NORM_FOUR_COLUMNS_FLAG, interrupt=True)

    def test_exact_mid_windows_keep_other_flags_fixed(self):
        result, _ = self.invoke(experiment_flag=windows.EXACT_MID_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.EXACT_MID_FLAG)
        self.assertTrue(all("exact_mid_flag" in row and "rank_flag" not in row
                            for row in result["window_samples"]))
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["exact_mid0_median_window_tps"], 1.0)
        self.assertEqual(summary["exact_mid1_median_window_tps"], 2.0)
        self.assertEqual(summary["geometric_mean_matched_window_ratio"], 2.0)

    def test_exact_wide_windows_keep_existing_dense_routes_fixed(self):
        result, _ = self.invoke(experiment_flag=windows.EXACT_WIDE_FLAG)
        self.assertEqual(result["window_protocol"]["experiment_flag"], windows.EXACT_WIDE_FLAG)
        self.assertTrue(all("exact_wide_flag" in row and "exact_mid_flag" not in row
                            for row in result["window_samples"]))
        summary = result["window_summary"]["32768"]
        self.assertEqual(summary["exact_wide0_median_window_tps"], 1.0)
        self.assertEqual(summary["exact_wide1_median_window_tps"], 2.0)
        self.assertEqual(summary["geometric_mean_matched_window_ratio"], 2.0)

    def test_exact_wide_interruption_restores_environment(self):
        with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
            self.invoke(experiment_flag=windows.EXACT_WIDE_FLAG, interrupt=True)
        self.assertNotIn("window_summary", self.last_result)

    def test_exact_expert_uses_four_columns_not_boolean_one(self):
        result, _ = self.invoke(experiment_flag=windows.EXACT_EXPERT_FLAG)
        self.assertEqual(result["window_protocol"]["enabled_value"], 4)
        self.assertEqual({row["exact_expert_flag"] for row in result["window_samples"]}, {0, 4})
        self.assertEqual(result["window_summary"]["32768"]["exact_expert4_median_window_tps"], 2.0)

    def test_branch_norm_keeps_boolean_protocol(self):
        result, _ = self.invoke(experiment_flag=windows.BRANCH_NORM_FLAG)
        self.assertEqual(result["window_protocol"]["enabled_value"], 1)
        self.assertEqual({row["branch_norm_flag"] for row in result["window_samples"]}, {0, 1})

    def test_large_cache_is_isolated_and_restored(self):
        result, _ = self.invoke(experiment_flag=windows.LARGE_CACHE_FLAG)
        self.assertEqual({row["large_cache_flag"] for row in result["window_samples"]}, {0, 1})
        self.assertEqual(result["window_summary"]["32768"]["large_cache1_median_window_tps"], 2.0)


if __name__ == "__main__":
    unittest.main()
