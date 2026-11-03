from pathlib import Path

from build_fc1_warp_fold import FLAG, main


PROGRAM = "QmoeFC1WarpFoldVec2BitcastF16"


if __name__ == "__main__":
    main(output=Path("/workspace/qwen38-int4-fc1-warp-fold/build-v3"), vector_fold=True)
