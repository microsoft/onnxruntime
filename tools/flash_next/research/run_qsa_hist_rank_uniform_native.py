from build_qsa_hist_rank import FLAG, HISTOGRAM, UNIFORM_PROGRAM
from qualify_quickgelu_mul_long_context import ROOT
from run_qsa_pruned_rank_native import main


if __name__ == "__main__":
    main(variant="qsa-hist-rank-v2", selector=FLAG, program_name=UNIFORM_PROGRAM,
         report_status="exact_reference_and_histogram_rank_dispatch_qualified",
         workgroup_size=64, pruning_enabled=True, requires_subgroups=False,
         parent_path=ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json",
         stamp="20261031", auxiliary_programs=(HISTOGRAM,), minimum_score_stride=8192)
