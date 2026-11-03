from build_qmoe_prefill1280 import FLAG
from run_prefill512_dsa_native import main


if __name__ == "__main__":
    main(variant="prefill1280-v2", fixture_version="v1", chunk=1280, selector=FLAG,
         extra_flags={"ORT_WEBGPU_QMOE_PREFILL1024": "1", "ORT_WEBGPU_DSA_QUERY_CACHE": "1"})
