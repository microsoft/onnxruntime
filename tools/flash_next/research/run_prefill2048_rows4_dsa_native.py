from pathlib import Path

from build_qmoe_prefill2048_rows4 import FLAG
from run_prefill512_dsa_native import main


if __name__ == "__main__":
    main(variant="prefill2048-rows4-v2", fixture_version="v1", chunk=2048, selector=FLAG,
         fixtures_directory=Path("/workspace/qwen38-int4-qmoe-prefill2048/fixtures-dsa-v1"),
         stamp="20261031",
         extra_flags={"ORT_WEBGPU_QMOE_PREFILL1024": "1", "ORT_WEBGPU_DSA_QUERY_CACHE": "1",
                      "ORT_WEBGPU_DENSE_BLOCK32": "1"})
