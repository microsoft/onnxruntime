import json
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


OUTPUT = Path("/workspace/qwen38-int4-fc2-static-cache/build-v1")
PARENT = Path("/workspace/qwen38-int4-fc2-private-a/build-v1")
FLAG = "ORT_WEBGPU_QMOE_FC2_STATIC_CACHE"


def transform(source):
    source = replace_once(source, "class QmoeFC2FinalMixProgram final", f'''namespace {{
bool Fc2StaticCacheEnabled() {{
  static const bool enabled = []() {{
    const int value = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(value == 0 || value == 1, "{FLAG} must be 0 or 1");
    ORT_ENFORCE(value == 0 || Fc2PrivateAEnabled(),
                "{FLAG} requires ORT_WEBGPU_QMOE_FC2_PRIVATE_A=1");
    return value == 1;
  }}();
  return enabled;
}}
}}

class QmoeFC2FinalMixProgram final''')
    source = replace_once(source, 'Program{Fc2PrivateAEnabled() && bitcast ?',
        'Program{Fc2StaticCacheEnabled() && bitcast ? "QmoeFC2StaticCacheBitcastF16" : Fc2PrivateAEnabled() && bitcast ?')
    source = replace_once(source, "private_a_{Fc2PrivateAEnabled() && bitcast} {}",
        "private_a_{Fc2PrivateAEnabled() && bitcast}, static_cache_{Fc2StaticCacheEnabled() && bitcast} {}")
    begin = source.index("    if (private_a_) {\n      code")
    end = source.index('    code          << "  for (var c', begin)
    original = source[begin:end]
    source = source[:begin] + '''    if (static_cache_) {
      for (int i = 0; i < 8; ++i) {
        code << "  let cached_a" << i << " = "
             << a.GetByOffset("rank * 160u + lane * 8u + " + std::to_string(i) + "u") << ";\\n";
      }
    } else ''' + original.lstrip() + source[end:]
    begin = source.index('         << "    for (var i = 0u; i < 4u;')
    end = source.index('         << "    partials[rank * 2u', begin)
    original = source[begin:end]
    source = source[:begin] + ''';
    if (static_cache_) {
      const char* components[] = {"x", "y", "z", "w"};
      for (int i = 0; i < 4; ++i) {
        code << "    {\\n"
             << "      let low = decode_low_nibbles(packed." << components[i] << ");\\n"
             << "      let high = decode_low_nibbles(packed." << components[i] << " >> 4u);\\n"
             << "      let w0 = vec4<f16>(low.x, high.x, low.y, high.y) * scale;\\n"
             << "      let w1 = vec4<f16>(low.z, high.z, low.w, high.w) * scale;\\n"
             << "      sum += dot(cached_a" << i * 2 << ", w0) + dot(cached_a" << i * 2 + 1 << ", w1);\\n"
             << "    }\\n";
      }
    } else {
      code ''' + original + ''';
    }
    code ''' + source[end:]
    return replace_once(source, "  bool private_a_;\n", "  bool private_a_;\n  bool static_cache_;\n")


if __name__ == "__main__":
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    if sha(PARENT / "libonnxruntime_providers_webgpu.so") != parent["provider_sha256"]:
        raise RuntimeError("Changed private-A parent")
    for p, digest in parent["source_sha256"].items():
        if sha(Path(p)) != digest:
            raise RuntimeError(f"Changed parent source: {p}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    working = Path(parent["working_directory"])
    library = OUTPUT / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=working, check=True)
    if sha(library) != parent["provider_sha256"]:
        raise RuntimeError("Parent relink mismatch")
    command = parent["compile_command"].copy()
    baseline = OUTPUT / "qmoe.baseline.cc.o"
    command[command.index("-o") + 1] = str(baseline)
    subprocess.run(command, cwd=working, check=True)
    if sha(baseline) != sha(PARENT / "qmoe.cc.o"):
        raise RuntimeError("Original QMoE object mismatch")
    candidate_include = OUTPUT / "qmoe_fc2_final_mix.inc"
    candidate_include.write_text(transform((PARENT / candidate_include.name).read_text()))
    candidate = OUTPUT / "qmoe.cc"
    candidate.write_text((PARENT / candidate.name).read_text())
    original_macro = next(c.split("=", 2)[2] for c in command if c.startswith("-fmacro-prefix-map="))
    command.insert(1, f"-fmacro-prefix-map={candidate}={original_macro}")
    command[command.index("-c") + 1] = str(candidate)
    obj = OUTPUT / "qmoe.cc.o"
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=working, check=True)
    old = str(PARENT / "qmoe.cc.o")
    if link.count(old) != 1:
        raise RuntimeError("Nonunique QMoE object replacement")
    link[link.index(old)] = str(obj)
    subprocess.run(link, cwd=working, check=True)
    sources = [Path(__file__), PARENT / "build-manifest.json", PARENT / "qmoe.cc",
        PARENT / "qmoe_fc2_final_mix.inc", candidate, candidate_include, obj, baseline]
    result = dict(status="fc2_static_cache_built", flag=FLAG, default_enabled=False,
        provider_sha256=sha(library), parent_provider_sha256=parent["provider_sha256"],
        parent_relink_byte_identical=True, baseline_qmoe_object_byte_identical=True,
        process_static_selector=True, workgroup_threads=320, declared_workgroup_bytes=1280,
        no_public_kernel_layout_changes=True, original_word_add_fold_and_rank_mix_preserved=True,
        physical_register_residency_verified=False, native_comparator=parent["native_comparator"],
        compile_command=command, link_command=link, working_directory=str(working),
        source_sha256={str(p): sha(p) for p in sources},
        model_eligible=False, performance_qualified=False, quality_qualified=False)
    (OUTPUT / "build-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "provider_sha256")}), flush=True)
