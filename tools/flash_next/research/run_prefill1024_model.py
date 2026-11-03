from build_qmoe_prefill1024 import FLAG
from run_prefill256_model import main


if __name__ == "__main__":
    main(variant="prefill1024-v2", stamp="20261030", parent_variant="prefill512-v3",
         parent_stamp="20261030", parent_chunk=512, selector=FLAG, chunk=1024,
         parent_standard_prefix="prefill512-v3-gpu7-full-standard2048-20261030",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-qmoe-prefill512/build-v3/libonnxruntime_providers_webgpu.so")
