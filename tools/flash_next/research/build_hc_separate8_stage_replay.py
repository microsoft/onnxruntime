import json
import os
from pathlib import Path

from build_hc_separate8 import transform as original_transform
from build_hc_projection_inner_unroll import SOURCE
from build_hc_staged_projection import REPLAY, stage_replay
from build_qmoe_node_replay import replace_once, sha
from build_scoped_webgpu_source import build


FLAG = "ORT_WEBGPU_HC_SEPARATE8_STAGE_REPLAY"
OUTPUT = Path("/workspace/qwen38-int4-hc-separate8-stage-replay/build-v1")
HEADER = OUTPUT.parent / "stage-replay-v1/hc_separate8_stage_replay.h"


def transform(source):
    result = original_transform(source)
    result = result.replace("ORT_WEBGPU_HC_SEPARATE8", FLAG)
    result = replace_once(result, '#include "static_node_replay.h"',
                          '#include "static_node_replay.h"\n' + f'#include "{HEADER}"')
    start = result.index("        if (separate8 == 1) {\n")
    end = result.index("        HCPostNormProjectionProgram program", start)
    branch = result[start:end]
    body = branch[len("        if (separate8 == 1) {\n"):]
    assert body.endswith("        }\n")
    body = body[:-len("        }\n")]
    marker = "    ORT_RETURN_IF_ERROR(context.RunProgram(norm));\n"
    norm, projection = body.split(marker)
    norm += "    return context.RunProgram(norm);\n"
    result = replace_once(result, branch, "")
    stages = (
        "    if (fused && separate8 == 1) {\n"
        "      ORT_RETURN_IF_ERROR(norm_replay_.Run(context, {y, normalized}, [&]() -> Status {\n" +
        norm + "      }));\n"
        "      return projection_replay_.Run(context, {activated, tail}, [&]() -> Status {\n" +
        projection + "      }, 0, 0, {normalized});\n"
        "    }\n")
    result = replace_once(result, "    if (fused) {\n", stages + "    if (fused) {\n")
    result = replace_once(result,
        "                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {",
        "                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}},\n"
        '        norm_replay_{shared ? "HCSharedSeparate8Norm" : "HCSeparate8Norm",\n'
        "                     shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5}\n"
        "                            : std::initializer_list<int>{0, 1, 2, 3}},\n"
        '        projection_replay_{shared ? "HCSharedSeparate8Projection" : "HCSeparate8Projection",\n'
        "                           shared ? std::initializer_list<int>{7, 8}\n"
        "                                  : std::initializer_list<int>{5, 6}} {")
    return replace_once(result, "  mutable StaticNodeReplay replay_;\n",
        "  mutable StaticNodeReplay replay_;\n"
        "  mutable StaticNodeReplay norm_replay_;\n"
        "  mutable HCSeparate8StageReplay projection_replay_;\n")


if __name__ == "__main__":
    parent_path = REPLAY.parent.parent / "build-manifest.json"
    parent = json.loads(parent_path.read_text())
    assert sha(REPLAY) == parent["source_sha256"]["overlay/static_node_replay.h"]
    HEADER.parent.mkdir(parents=True, exist_ok=False)
    HEADER.write_text(stage_replay(REPLAY.read_text()).replace("HCStageReplay", "HCSeparate8StageReplay"))
    report = build(OUTPUT, SOURCE, transform, SOURCE.parent / "native_compare.cpp", FLAG,
        dict(model_eligible=False, normalization_workgroups=4, normalization_threads=256,
             projection_workgroups=41, projection_threads=256, output_columns=8,
             projection_shared_storage_bytes=512,
             original_rms_streams=256, original_rms_tree_preserved=True,
             original_fp16_projection_schedule_preserved=True,
             each_stage_captures_one_command=True,
             normalized_input_in_projection_key_and_binding=True,
             no_additional_persistent_gpu_buffers=True,
             original_replay_implementation_unchanged=True))
    report.update(original_compute_and_fallbacks_unchanged=False,
                  original_default_compute_and_fallbacks_unchanged=True,
                  no_object_layout_changes=False, private_leaf_kernel_layout_changed=True,
                  no_runtime_object_layout_changes=True)
    report["source_sha256"]["../stage-replay-v1/" + HEADER.name] = sha(HEADER)
    report["source_sha256"][os.path.relpath(parent_path, OUTPUT)] = sha(parent_path)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
