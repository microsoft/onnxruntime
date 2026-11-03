from build_qmoe_prefill512 import FLAG
from run_prefill256_model import main


if __name__ == "__main__":
    main(variant="prefill512-v3", stamp="20261030", parent_variant="dense-tile4-v1",
         parent_stamp="20261030", parent_chunk=256, selector=FLAG, chunk=512)
