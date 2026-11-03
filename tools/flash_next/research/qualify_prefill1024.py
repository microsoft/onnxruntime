from qualify_prefill256 import main


if __name__ == "__main__":
    main(variant="prefill1024-v2", stamp="20261030", parent_variant="prefill512-v3",
         parent_stamp="20261030", previous_chunk=512, chunk=1024)
