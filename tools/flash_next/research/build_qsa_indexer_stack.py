from pathlib import Path

from build_qmoe_node_replay import replace_once
from build_qsa_concat_vector import main as build_leaf, transform as concat_transform
from build_qsa_packed_score import transform as score_transform


OUTPUT = Path("/workspace/qwen38-int4-qsa-indexer-stack/build-v1")
FLAG = "ORT_WEBGPU_QSA_INDEXER_STACK"


def transform(source):
    source = score_transform(concat_transform(source))
    anchor = '  const int concat_vector = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_CONCAT_VECTOR", 0);'
    selector = f"""  const int indexer_stack = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(indexer_stack == 0 || indexer_stack == 1, "{FLAG} must be 0 or 1");
"""
    source = replace_once(source, anchor, selector + anchor)
    source = replace_once(source, "if (concat_vector == 1 && head_size % 4 == 0",
                          "if ((concat_vector == 1 || indexer_stack == 1) && head_size % 4 == 0")
    return replace_once(source, "if (packed_score == 1 && use_global_cache && rows == 1",
                        "if ((packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1")


if __name__ == "__main__":
    build_leaf(output=OUTPUT, source_transform=transform)
