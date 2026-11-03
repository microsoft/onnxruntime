from build_hc_post_norm_pad34 import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="hc-post-norm-pad34-v1", experiment_flag=FLAG, stamp="20261030",
         parent_variant="hc-up-parallel2-v1", program_name="HCPostNormProjectionPad34F16",
         original_program_name="HCPostNormProjectionF16", dispatches=94)
