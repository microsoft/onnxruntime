import collections
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from build_gdn_tree32_hist import OUTPUT, stack_link
import qualify_gdn_tree32
from run_hc_up_parallel2_model import verify_gdn_histogram_profile


class GdnTree32HistogramTests(unittest.TestCase):
    def setUp(self):
        base = Path("/home/sayanshaw/repos")
        self.histogram = json.loads((base / "qwen38-int4-qsa-hist-rank/build-v2/build-manifest.json").read_text())
        self.tree = json.loads((base / "qwen38-int4-gdn-tree32/build-v1/build-manifest.json").read_text())

    def test_real_inputs_replace_only_gdn_and_output(self):
        link = stack_link(self.histogram, self.tree)
        changes = [(before, after) for before, after in zip(self.histogram["link_command"], link)
                   if before != after]
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[1][1], str(OUTPUT / "gated_delta_net.cc.o"))
        self.assertIn("/workspace/qwen38-int4-qsa-hist-rank/build-v2/sparse_attention_indexer.cc.o", link)

    def test_parent_extra_object_reordering_and_missing_input_rejected(self):
        for kind in ("parent", "object", "order", "missing"):
            histogram = copy.deepcopy(self.histogram)
            if kind == "parent":
                histogram["baseline_provider_sha256"] = "different"
            elif kind == "object":
                histogram["link_command"][-1] = "unqualified.o"
            elif kind == "order":
                histogram["link_command"].reverse()
            else:
                histogram["link_command"].pop()
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                stack_link(histogram, self.tree)

    def test_both_histogram_programs_and_original_fallback_checked(self):
        for context in (2048, 8192, 32768):
            expected = 12 if context == 32768 else 0
            counts = collections.Counter(QsaHistogramUniformRank=expected, QsaRankHistogram=expected,
                                         SparseAttentionIndexerQsaParallelRank=0 if expected else 12)
            self.assertEqual(verify_gdn_histogram_profile(counts, context),
                             {"QsaHistogramUniformRank": expected, "QsaRankHistogram": expected})
            for name in counts:
                changed = counts.copy()
                changed[name] += 1
                with self.subTest(context=context, name=name), self.assertRaises(AssertionError):
                    verify_gdn_histogram_profile(changed, context)

    def test_unqualified_variant_or_wrong_parent_rejected_before_reading_evidence(self):
        with patch("sys.argv", ["qualifier"]):
            for variant, parent in (("unknown-v1", "qsa-hist-rank-v2"),
                                    ("gdn-tree32-hist-v1", "qsa-prefill-score16-v1"),
                                    ("gdn-tree32-v1", "qsa-hist-rank-v2")):
                with self.subTest(variant=variant, parent=parent), self.assertRaises(AssertionError):
                    qualify_gdn_tree32.main(variant=variant, parent_variant=parent)

    def test_stacked_standard_requires_both_flags_and_explicit_chunk(self):
        raw = {"load_experiment_flags": {"ORT_WEBGPU_GDN_TREE32": "1", "ORT_WEBGPU_QSA_HIST_RANK": "1"}}
        with patch.object(qualify_gdn_tree32, "verify_standard", return_value="verified") as verifier:
            self.assertEqual(qualify_gdn_tree32.verify_gdn_histogram_standard(
                raw, {}, {}, {}, [], 32768, False), "verified")
            self.assertEqual(verifier.call_args.kwargs, {"chunk": 1024})
            for flag in raw["load_experiment_flags"]:
                changed = copy.deepcopy(raw)
                changed["load_experiment_flags"][flag] = "0"
                with self.subTest(flag=flag), self.assertRaises(AssertionError):
                    qualify_gdn_tree32.verify_gdn_histogram_standard(changed, {}, {}, {}, [], 32768, False)


if __name__ == "__main__":
    unittest.main()
