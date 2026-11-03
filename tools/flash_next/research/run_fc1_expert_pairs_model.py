from build_fc1_expert_pairs import FLAG, PROGRAM
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="fc1-expert-pairs-v1", experiment_flag=FLAG, stamp="20261031",
         parent_variant="qmoe-prefill-rows4-v2", parent_stamp="20261031", chunk=1024,
         parent_standard_prefix="qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                         "libonnxruntime_providers_webgpu.so",
         program_name=PROGRAM, original_program_name="QmoeFC1SwiGLUBitcastF16", dispatches=48)
