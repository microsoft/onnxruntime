from pathlib import Path

from build_qsa_concat_vector import main as build
from build_qsa_hist_rank import PARENT, transform


if __name__ == "__main__":
    build(output=Path("/workspace/qwen38-int4-qsa-hist-rank/build-v2"),
          source_transform=lambda source: transform(source, minimum_stride=8192, uniform_exit=True),
          parent_path=PARENT, component_path=PARENT, no_arithmetic_changes=False)
