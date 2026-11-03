from qualify_prefill256 import main


if __name__ == "__main__":
    main(variant="prefill512-v3", stamp="20261030", parent_variant="dense-tile4-v1",
         parent_stamp="20261030", previous_chunk=256, chunk=512)
