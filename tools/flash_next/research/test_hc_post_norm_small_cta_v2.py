from pathlib import Path
import unittest

from build_hc_post_norm_small_cta_v2 import candidate_source, private_replay


class SmallCtaPrivateReplayTests(unittest.TestCase):
    def test_only_private_type_variant_argument_and_key_change(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-static-node-replay/"
                      "build-v8/overlay/static_node_replay.h").read_text()
        result = private_replay(source)
        restored = result.replace("HcSmallCtaStaticNodeReplay", "StaticNodeReplay")
        restored = restored.replace(", int64_t variant = 0", "")
        restored = restored.replace("\n    key.push_back(variant);", "")
        self.assertEqual(restored, source)
        self.assertIn("std::lock_guard<std::mutex> lock(mutex_);", result)

    def test_private_embedding_and_compile_time_size_check(self):
        source = Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/"
                      "build-v2/hc_post_norm_projection.cc").read_text()
        result = candidate_source(source)
        self.assertIn('#include "hc_small_cta_replay.h"', result)
        self.assertIn("mutable HcSmallCtaStaticNodeReplay replay_;", result)
        self.assertIn("static_assert(sizeof(HcSmallCtaStaticNodeReplay) == sizeof(StaticNodeReplay));", result)
        self.assertNotIn("mutable StaticNodeReplay replay_;", result)
        self.assertIn("}, 0, small_cta);", result)


if __name__ == "__main__":
    unittest.main()
