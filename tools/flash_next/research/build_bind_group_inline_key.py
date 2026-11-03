import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_BIND_GROUP_INLINE_KEY"
OUTPUT = Path("/workspace/qwen38-int4-bind-group-inline-key/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-static-node-replay/build-v8")
SOURCE = COMPONENT / "overlay/core/providers/webgpu/webgpu_context.cc"
HELPERS = f'''namespace {{
bool BindGroupInlineKeyEnabled() {{
  static const bool enabled = []() {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    return value == 1;
  }}();
  return enabled;
}}

std::span<const uint64_t> BindGroupLookupKey(
    const WGPUBindGroupDescriptor& descriptor, bool inline_key,
    std::array<uint64_t, 129>& local, std::vector<uint64_t>& owned) {{
  if (inline_key && descriptor.entryCount <= 32) {{
    local[0] = reinterpret_cast<uintptr_t>(descriptor.layout);
    size_t index = 1;
    for (size_t i = 0; i < descriptor.entryCount; ++i) {{
      const auto& entry = descriptor.entries[i];
      local[index++] = entry.binding;
      local[index++] = reinterpret_cast<uintptr_t>(entry.buffer);
      local[index++] = entry.offset;
      local[index++] = entry.size;
    }}
    return {{local.data(), index}};
  }}
  owned.reserve(1 + descriptor.entryCount * 4);
  owned.push_back(reinterpret_cast<uintptr_t>(descriptor.layout));
  for (size_t i = 0; i < descriptor.entryCount; ++i) {{
    const auto& entry = descriptor.entries[i];
    owned.insert(owned.end(), {{entry.binding, reinterpret_cast<uintptr_t>(entry.buffer), entry.offset, entry.size}});
  }}
  return {{owned.data(), owned.size()}};
}}
}}  // namespace

'''
ORIGINAL_HASH = '''  struct Hash {
    size_t operator()(const std::vector<uint64_t>& key) const noexcept {
      size_t result = 0;
      for (uint64_t value : key) {
        result ^= std::hash<uint64_t>{}(value) + size_t{0x9e3779b9} + (result << 6) + (result >> 2);
      }
      return result;
    }
  };'''
HASH = ORIGINAL_HASH.replace("    size_t operator()", "    using is_transparent = void;\n    size_t operator()")
HASH = HASH.replace("const std::vector<uint64_t>& key", "std::span<const uint64_t> key")
HASH += '''
  struct Equal {
    using is_transparent = void;
    bool operator()(std::span<const uint64_t> left, std::span<const uint64_t> right) const noexcept {
      return std::equal(left.begin(), left.end(), right.begin(), right.end());
    }
  };'''
ORIGINAL_KEY = '''  std::vector<uint64_t> key;
  key.reserve(1 + descriptor.entryCount * 4);
  key.push_back(reinterpret_cast<uintptr_t>(descriptor.layout));
  for (size_t i = 0; i < descriptor.entryCount; ++i) {
    const auto& entry = descriptor.entries[i];
    key.insert(key.end(), {entry.binding, reinterpret_cast<uintptr_t>(entry.buffer), entry.offset, entry.size});
  }'''


def transform(source):
    source = source.replace("#include", "#include <array>\n#include <span>\n#include", 1)
    source = replace_once(source, "struct WebGpuContextFactory::BindGroupCache {",
                          HELPERS + "struct WebGpuContextFactory::BindGroupCache {")
    source = replace_once(source, ORIGINAL_HASH, HASH)
    source = replace_once(source,
        "std::unordered_map<std::vector<uint64_t>, wgpu::BindGroup, Hash> entries;",
        "std::unordered_map<std::vector<uint64_t>, wgpu::BindGroup, Hash, Equal> entries;")
    source = replace_once(source, ORIGINAL_KEY, '''  std::array<uint64_t, 129> local_key;
  std::vector<uint64_t> key;
  const auto lookup_key = BindGroupLookupKey(descriptor, BindGroupInlineKeyEnabled(), local_key, key);''')
    source = replace_once(source, "auto found = cache.entries.find(key);", "auto found = cache.entries.find(lookup_key);")
    return replace_once(source, "  cache.entries.emplace(std::move(key), result);",
        "  if (key.empty()) key.assign(lookup_key.begin(), lookup_key.end());\n"
        "  cache.entries.emplace(std::move(key), result);")


def main():
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    for name, digest in component["overlay_source_sha256"].items():
        if sha(COMPONENT / "overlay" / name) != digest:
            raise RuntimeError(f"Changed retained source: {name}")
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
    result = dict(status="bind_group_inline_key_built", flag=FLAG, default_enabled=False,
        process_static_selector=True, stack_binding_capacity=32, oversized_key_original_heap_fallback=True,
        original_context_object_byte_identical=True, parent_relink_byte_identical=True,
        no_public_kernel_layout_changes=True, cache_entry_and_byte_limits_unchanged=True,
        full_key_equality_and_owned_insert_preserved=True, no_shader_or_arithmetic_changes=True,
        provider_sha256=sha(library), parent_provider_sha256=parent["provider_sha256"],
        compile_command=command, link_command=link, working_directory=str(working),
        source_sha256={str(path): sha(path) for path in sources},
        model_eligible=False, performance_qualified=False, quality_qualified=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "provider_sha256")}), flush=True)


if __name__ == "__main__":
    main()
