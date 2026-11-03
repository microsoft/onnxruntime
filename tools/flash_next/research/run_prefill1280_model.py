from build_qmoe_prefill1280 import FLAG
from run_prefill256_model import main


if __name__ == "__main__":
    main(variant="prefill1280-v2", stamp="20261030", parent_variant="dsa-query-cache-v2",
         parent_stamp="20261030", parent_chunk=1024, selector=FLAG, chunk=1280,
         parent_standard_prefix="dsa-query-cache-v2-chunk1024-gpu7-full-standard2048-20261030",
         parent_provider="/home/sayanshaw/repos/qwen38-int4-dsa-query-cache/build-v2/libonnxruntime_providers_webgpu.so")
