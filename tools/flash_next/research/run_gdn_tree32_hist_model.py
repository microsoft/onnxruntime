from build_gdn_tree32 import FLAG, PROGRAM
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="gdn-tree32-hist-v1", experiment_flag=FLAG, stamp="20261031",
         parent_variant="qsa-hist-rank-v2", parent_stamp="20261031", chunk=1024,
         parent_standard_prefix="qsa-hist-rank-v2-chunk1024-gpu7-full-standard2048-20261031",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-hist-rank/build-v2/"
                         "libonnxruntime_providers_webgpu.so",
         program_name=PROGRAM, original_program_name="GatedDeltaNet", dispatches=36)
