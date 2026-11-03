from build_dense_tile4 import FLAG
from qualify_hc_up_parallel2_windows import main


if __name__ == "__main__":
    main(variant="dense-tile4-v1", experiment_flag=FLAG, stamp="20261030",
         parent_variant="hc-up-parallel2-v1", program_name="MatMulNBitsTile4F16",
         original_program_name="MatMulNBitsBitcastF16", dispatches=159)
