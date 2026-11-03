from build_qmoe_prefill2048 import FLAG
from run_prefill256_model import main


if __name__ == "__main__":
    main(variant="prefill2048-v2", stamp="20261030", parent_variant="dense-block32-v2",
         parent_stamp="20261030", parent_chunk=1024, selector=FLAG, chunk=2048)
