from build_dense_static_shape import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="dense-static-shape-v1", experiment_flag=FLAG, stamp="20261030",
         parent_variant="dense-block32-v2", parent_stamp="20261030", chunk=1024,
         program_name="MatMulNBitsStaticShapeF16", original_program_name="MatMulNBitsBlock32F16",
         dispatches=159)
