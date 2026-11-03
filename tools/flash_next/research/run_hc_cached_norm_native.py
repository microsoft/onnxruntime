from pathlib import Path
from functools import partial

from qualify_quickgelu_mul_long_context import ROOT
from run_hc_bitcast_native import main, verify_static_stats


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-hc-cached-norm/build-v2"),
         prefix=ROOT / "hc-cached-norm-v2-native-gpu7-20261031",
         report_status="hc_cached_norm_four_outputs_finite_bitwise_and_compound_dispatch_qualified",
         parent_path=ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json",
         stats_verifier=partial(verify_static_stats, requests=96), compound_candidate_not_replay_cached=True)
