from build_hc_post_norm_predecode import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="hc-post-norm-predecode-v2", experiment_flag=FLAG, stamp="20261030",
         parent_variant="dense-block32-v2", parent_stamp="20261030", chunk=1024,
         program_name="HCPostNormProjectionPredecodedF16", original_program_name="HCPostNormProjectionF16",
         dispatches=94)
