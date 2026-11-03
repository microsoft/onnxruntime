import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-gpu-slice-copy/build-v1")
BASE = Path("/workspace/qwen38-int4-gpu-greedy-argmax/build-v6")
DENSE = Path("/workspace/qwen38-int4-dense-columns8-double-buffer/build-v1")
SOURCE = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/core/providers/webgpu/tensor/slice.cc")


def transform(source):
    source = replace_once(source, '#include "core/common/inlined_containers.h"',
                          '#include "core/common/inlined_containers.h"\n#include "core/platform/env_var_utils.h"')
    source = replace_once(source, "namespace webgpu {", """namespace webgpu {

namespace {
const std::vector<MLDataType>& SliceCopyTypes() {
  static const std::vector<MLDataType> types = []() {
    auto result = WebGpuSupportedFloatTypes();
    const int selected = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_SLICE_UINT32_BITCOPY", 0);
    ORT_ENFORCE(selected == 0 || selected == 1, "ORT_WEBGPU_SLICE_UINT32_BITCOPY must be 0 or 1");
    if (selected == 1)
      result.push_back(DataTypeImpl::GetTensorType<uint32_t>());
    return result;
  }();
  return types;
}
}  // namespace
""")
    return replace_once(source,
        '    13,\n    kWebGpuExecutionProvider,\n    (*KernelDefBuilder::Create())\n        .TypeConstraint("T", WebGpuSupportedFloatTypes())',
        '    13,\n    kWebGpuExecutionProvider,\n    (*KernelDefBuilder::Create())\n        .TypeConstraint("T", SliceCopyTypes())')


if __name__ == "__main__":
    manifest = json.loads((BASE / "build-manifest.json").read_text())
    dense = json.loads((DENSE / "build-manifest.json").read_text())
    if sha(BASE / "libonnxruntime_providers_webgpu.so") != manifest["provider_sha256"]:
        raise RuntimeError("Changed composite provider")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    working = Path(json.loads(Path("/workspace/qwen38-int4-qmoe-prefill-rows4/build-v2/build-manifest.json").read_text())["working_directory"])
    link = manifest["provider_link_command"].copy()
    old = next(x for x in link if x.endswith("core__providers__webgpu__tensor__slice.cc.o"))
    provider = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link[link.index("-o") + 1] = str(provider)
    subprocess.run(link, cwd=working, check=True)
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError("Composite provider relink mismatch")
    compile_command = dense["compile_command"].copy()
    baseline = OUTPUT / "slice.baseline.cc.o"
    compile_command[compile_command.index("-c") + 1] = str(SOURCE)
    compile_command[compile_command.index("-o") + 1] = str(baseline)
    subprocess.run(compile_command, cwd=working, check=True)
    if sha(baseline) != sha(Path(old)):
        raise RuntimeError("Original Slice object not byte-identical")
    candidate = OUTPUT / "slice.cc"
    candidate.write_text(transform(SOURCE.read_text()))
    obj = OUTPUT / "slice.cc.o"
    compile_command[compile_command.index("-c") + 1] = str(candidate)
    compile_command[compile_command.index("-o") + 1] = str(obj)
    compile_command.insert(1, f"-fmacro-prefix-map={candidate}={SOURCE}")
    subprocess.run(compile_command, cwd=working, check=True)
    link[link.index(old)] = str(obj)
    subprocess.run(link, cwd=working, check=True)
    result = dict(status="gated_uint32_slice_provider_built", default_enabled=False,
                  baseline_relink_byte_identical=True, baseline_slice_object_byte_identical=True,
                  provider_sha256=sha(provider), baseline_provider_sha256=manifest["provider_sha256"],
                  compile_command=compile_command, link_command=link,
                  source_sha256={str(p): sha(p) for p in
                      (Path(__file__), SOURCE, candidate, obj, baseline, BASE / "build-manifest.json", DENSE / "build-manifest.json")})
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("compile_command", "link_command", "source_sha256")}), flush=True)
