from build_hc_post_norm_wide16 import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="hc-post-norm-wide16-v1", experiment_flag=FLAG, stamp="20261031",
         parent_variant="dense-block32-v2", parent_stamp="20261030", chunk=1024,
         parent_standard_prefix="dense-block32-v2-chunk1024-gpu7-full-standard2048-20261030",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v2/libonnxruntime_providers_webgpu.so",
         program_name="HCPostNormProjectionWide16F16", original_program_name="HCPostNormProjectionF16",
         dispatches=94)
