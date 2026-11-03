from build_dsa_paired import FLAG
from run_dsa_query_cache_native import main


if __name__ == "__main__":
    main(variant="dsa-paired-v1", parent_variant="dense-block32-v2", flag=FLAG,
         program_name="DynamicSparseAttentionPaired",
         extra_flags={"ORT_WEBGPU_DSA_QUERY_CACHE": "1", "ORT_WEBGPU_DENSE_BLOCK32": "1"},
         report_status="dsa_paired_finite_bitwise_and_cpu_reference_qualified")
