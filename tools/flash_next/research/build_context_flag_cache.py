import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_STATIC_CONTEXT_FLAGS"
OUTPUT = Path("/workspace/qwen38-int4-context-flag-cache/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
SOURCE = COMPONENT / "overlay/core/providers/webgpu/webgpu_context.cc"
HELPERS = f'''namespace {{
bool StaticContextFlagsEnabled() {{
  static const bool enabled = []() {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    return value == 1;
  }}();
  return enabled;
}}

int ContextBindGroupCacheCapacity() {{
  auto read = []() {{
    return ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", 0);
  }};
  if (!StaticContextFlagsEnabled()) return read();
  static const int value = read();
  return value;
}}

int ContextDynamicUniforms() {{
  auto read = []() {{
    return ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_DYNAMIC_UNIFORMS", 0);
  }};
  if (!StaticContextFlagsEnabled()) return read();
  static const int value = read();
  return value;
}}
}}  // namespace

'''


def transform(source):
    for name, helper, count in (
        ("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", "ContextBindGroupCacheCapacity", 2),
        ("ORT_WEBGPU_DYNAMIC_UNIFORMS", "ContextDynamicUniforms", 1),
    ):
        original = f'ParseEnvironmentVariableWithDefault<int>("{name}", 0)'
        if source.count(original) != count:
            raise RuntimeError(f"Unexpected selector use count: {name}")
        source = source.replace(original, helper + "()")
    return replace_once(source, "Status WebGpuContext::PrepareNodeBinding(",
                        HELPERS + "Status WebGpuContext::PrepareNodeBinding(")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    for name, digest in component["overlay_source_sha256"].items():
        if sha(COMPONENT / "overlay" / name) != digest:
            raise RuntimeError(f"Changed retained context source: {name}")
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed composite parent")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    working = Path(parent["working_directory"])
    subprocess.run(link, cwd=working, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Composite parent relink mismatch")
    command = next(c.copy() for c in component["compile_commands"] if c[c.index("-c") + 1] == str(SOURCE))
    baseline = OUTPUT / "webgpu_context.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline)
    subprocess.run(command, cwd=working, check=True)
    original_object = COMPONENT / "webgpu_context.cc.o"
    if sha(baseline) != sha(original_object):
        raise RuntimeError("Original context object not byte-identical")
    candidate = OUTPUT / "webgpu_context.cc"
    candidate.write_text(transform(SOURCE.read_text()))
    obj = OUTPUT / "webgpu_context.cc.o"
    command[command.index("-c") + 1] = str(candidate)
    command[command.index("-o") + 1] = str(obj)
    command.insert(1, f"-fmacro-prefix-map={candidate}={SOURCE}")
    subprocess.run(command, cwd=working, check=True)
    if link.count(str(original_object)) != 1:
        raise RuntimeError("Nonunique context link object")
    link[link.index(str(original_object))] = str(obj)
    subprocess.run(link, cwd=working, check=True)
    sources = (Path(__file__), PARENT / "build-manifest.json", COMPONENT / "build-manifest.json",
               SOURCE, candidate, baseline, original_object, obj)
    result = dict(status="context_flag_cache_built", flag=FLAG, default_enabled=False,
        process_static_selector=True, selected_values_frozen_per_process_on_first_use=True,
        dynamic_reads_preserved_when_disabled=True, original_context_object_byte_identical=True,
        parent_relink_byte_identical=True, no_public_kernel_layout_changes=True,
        no_shader_or_arithmetic_changes=True, provider_sha256=sha(library),
        parent_provider_sha256=parent["provider_sha256"], compile_command=command, link_command=link,
        working_directory=str(working), source_sha256={str(p): sha(p) for p in sources},
        model_eligible=False, performance_qualified=False, quality_qualified=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "provider_sha256")}), flush=True)


if __name__ == "__main__":
    main()
