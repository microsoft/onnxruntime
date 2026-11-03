from build_qsa_subgroup_rank import FLAG
from run_qsa_pruned_rank_native import main


if __name__ == "__main__":
    main(variant="qsa-subgroup-rank-v1", selector=FLAG, program_name="QsaSubgroupRank",
         report_status="exact_reference_and_subgroup_rank_dispatch_qualified",
         workgroup_size=64, pruning_enabled=False)
