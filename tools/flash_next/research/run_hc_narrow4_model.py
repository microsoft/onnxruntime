from build_hc_narrow4 import FLAG, PROGRAM
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="hc-narrow4-v1", experiment_flag=FLAG, stamp="20261031",
         parent_variant="qsa-prefill-score16-v1", parent_stamp="20261031", chunk=1024,
         parent_standard_prefix="qsa-prefill-score16-v1-chunk1024-gpu7-full-standard2048-20261031",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/"
                         "libonnxruntime_providers_webgpu.so",
         program_name=PROGRAM, original_program_name="HCPostNormProjectionF16", dispatches=94)
