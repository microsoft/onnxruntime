from pathlib import Path

from prepare_prefill1280_dsa import main


if __name__ == "__main__":
    main(directory=Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill2048/fixtures-dsa-v1"),
         sequences=(2048,), boundary=2048)
