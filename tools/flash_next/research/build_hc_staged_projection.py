import json
from pathlib import Path

from build_hc_bitcast_unpack import main as build
from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_HC_STAGED_PROJECTION"
OUTPUT = Path("/workspace/qwen38-int4-hc-staged-projection/build-v1")
REPLAY = Path("/workspace/qwen38-int4-dense-bitcast-unpack/build-v1/overlay/static_node_replay.h")


def stage_replay(source):
    source = source.replace("StaticNodeReplay", "HCStageReplay")
    source = replace_once(source, "int64_t row_elements = 0, uint64_t variant = 0) {",
                          "int64_t row_elements = 0, uint64_t variant = 0,\n"
                          "             std::initializer_list<const Tensor*> extra_inputs = {}) {")
    source = replace_once(source, "    key.push_back(outputs.size());",
                          "    key.push_back(extra_inputs.size());\n"
                          "    for (const auto* input : extra_inputs) append(input, false);\n"
                          "    key.push_back(outputs.size());")
    marker = "      for (int index : gpu_inputs_) {"
    source = replace_once(source, marker, '''      for (const auto* tensor : extra_inputs) {
        ORT_ENFORCE(tensor != nullptr && tensor->Location().device.Type() == OrtDevice::GPU &&
                        tensor->Location().mem_type == OrtMemType::OrtMemTypeDefault &&
                        tensor->Location().name == WEBGPU_BUFFER,
                    "HC stage replay requires default WebGPU-buffer extra inputs.");
        buffers.push_back(reinterpret_cast<WGPUBuffer>(const_cast<void*>(tensor->DataRaw())));
      }
''' + marker)
    return source


def private_programs(source):
    start = source.index("class HCPostNormProjectionBaselineNormProgram")
    stop = source.index("class HCPostNormProjectionProgram final", start)
    return (source[start:stop]
            .replace("HCPostNormProjectionBaselineNormProgram", "HCStagedNormProgram")
            .replace('Program{"HCPostNormF16"}', 'Program{"HCStagedPostNormF16"}')
            .replace("HCPostNormProjectionBaselineMatMulProgram", "HCStagedProjectionProgram")
            .replace('Program{"HCProjectionSiLUF16"}', 'Program{"HCStagedProjectionF16"}'))


def transform(source):
    programs = private_programs(source)
    source = replace_once(source, '#include "static_node_replay.h"',
                          '#include "static_node_replay.h"\n#include "hc_stage_replay.h"')
    source = replace_once(source, "class HCPostNormProjectionF16",
                          programs + "class HCPostNormProjectionF16")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {",
        f'''  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override {{
    const int staged = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(staged == 0 || staged == 1, "{FLAG} must be 0 or 1");''')
    start = source.index("    HCPostNormProjectionBaselineNormProgram norm{")
    stop = source.index("    auto intermediate =", start)
    block = source[start:stop]
    norm, projection = block.split("    if (projection_fused) {\n", 1)
    norm = norm.replace("HCPostNormProjectionBaselineNormProgram", "HCStagedNormProgram")
    norm = replace_once(norm, "    ORT_RETURN_IF_ERROR(context.RunProgram(norm));",
                        "    return context.RunProgram(norm);")
    projection = projection.replace("HCPostNormProjectionBaselineMatMulProgram", "HCStagedProjectionProgram")
    assert projection.endswith("    }\n")
    projection = projection[:-6]
    branch = (
        "    if (fused && staged == 1) {\n"
        "      ORT_RETURN_IF_ERROR(stage_norm_.Run(context, {y, normalized}, [&]() -> Status {\n" +
        norm + "\n      }));\n"
        "      return stage_projection_.Run(context, {activated, tail}, [&]() -> Status {\n" +
        projection + "      }, 0, 0, {normalized});\n"
        "    }\n")
    source = replace_once(source, "    if (fused) {\n", branch + "    if (fused) {\n")
    source = replace_once(source,
        "                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}} {",
        '''                       : std::initializer_list<int>{0, 1, 2, 3, 5, 6}},
        stage_norm_{shared ? "HCSharedStagedNorm" : "HCStagedNorm",
                    shared ? std::initializer_list<int>{0, 1, 2, 3, 4, 5}
                           : std::initializer_list<int>{0, 1, 2, 3}},
        stage_projection_{shared ? "HCSharedStagedProjection" : "HCStagedProjection",
                          shared ? std::initializer_list<int>{7, 8}
                                 : std::initializer_list<int>{5, 6}} {''')
    return replace_once(source, "  mutable StaticNodeReplay replay_;\n",
                        "  mutable StaticNodeReplay replay_;\n"
                        "  mutable HCStageReplay stage_norm_;\n"
                        "  mutable HCStageReplay stage_projection_;\n")


def main():
    replay_manifest = json.loads((REPLAY.parent.parent / "build-manifest.json").read_text())
    assert sha(REPLAY) == replay_manifest["source_sha256"]["overlay/static_node_replay.h"]
    # The builder owns creation of OUTPUT; the include lives in a separate immutable directory.
    includes = OUTPUT.parent / "stage-replay-v1"
    includes.mkdir(parents=True, exist_ok=False)
    header = includes / "hc_stage_replay.h"
    header.write_text(stage_replay(REPLAY.read_text()))
    build(output=OUTPUT, flag=FLAG, source_transform=transform,
          parent_dir=Path("/workspace/qwen38-int4-qmoe-prefill512/build-v3"),
          extra_include_dirs=(includes,))
    path = OUTPUT / "build-manifest.json"
    report = json.loads(path.read_text())
    report.update(no_existing_kernel_object_layout_changes=False,
                  private_leaf_kernel_layout_changed=True, no_runtime_object_layout_changes=True,
                  frozen_stage_replay_parent_sha256=sha(REPLAY))
    report["source_sha256"]["../stage-replay-v1/hc_stage_replay.h"] = sha(header)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
