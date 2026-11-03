from build_dense_tile2 import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="dense-tile2-v2", experiment_flag=FLAG, stamp="20261030",
         parent_variant="prefill512-v3", parent_stamp="20261030", chunk=512,
         parent_standard_prefix="prefill512-v3-gpu7-full-standard2048-20261030",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3/libonnxruntime_providers_webgpu.so",
         program_name="MatMulNBitsTile2F16",
         original_program_name="MatMulNBitsTile4F16", dispatches=159)
