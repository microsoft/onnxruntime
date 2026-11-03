from build_qmoe_prefill1024 import FLAG
from run_prefill512_dsa_native import main


if __name__ == "__main__":
    main(variant="prefill1024-v2", fixture_version="v1", chunk=1024, selector=FLAG)
