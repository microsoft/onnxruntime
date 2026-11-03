from build_gdn_value2 import FLAG
from run_hc_up_parallel2_model import main


if __name__ == "__main__":
    main(variant="gdn-value2-v1", experiment_flag=FLAG, stamp="20261030",
         parent_variant="prefill1024-v2", parent_stamp="20261030", chunk=1024,
         parent_standard_prefix="prefill1024-v2-gpu7-full-standard2048-20261030",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-qmoe-prefill1024/build-v2/libonnxruntime_providers_webgpu.so",
         program_name="GatedDeltaNetValue2", original_program_name="GatedDeltaNet", dispatches=36)
