import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


FLAG = "ORT_WEBGPU_GDN_VALUE2"
PARENT = Path("/workspace/qwen38-int4-qmoe-prefill1024/build-v2")
COMPONENT = Path("/workspace/qwen38-int4-gdn-decode-view/build-original")
HEADER = Path("/workspace/onnxruntime-webgpu-h200-perf/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.h")
OUTPUT = Path("/workspace/qwen38-int4-gdn-value2/build-v1")


def transform(source, header):
    source = replace_once(source, "#include <limits>", "#include <limits>\n#include <string_view>")
    start = header.index("class GatedDeltaNetProgram final")
    stop = header.index("\n};", start) + len("\n};")
    declaration = header[start:stop].replace("GatedDeltaNetProgram", "GatedDeltaNetValue2Program")
    declaration = replace_once(declaration, 'Program{"GatedDeltaNet"}', 'Program{"GatedDeltaNetValue2"}')
    start = source.index("Status GatedDeltaNetProgram::GenerateShaderCode")
    stop = source.index("Status GatedDeltaNetNormalizeProgram::GenerateShaderCode", start)
    method = source[start:stop].replace("GatedDeltaNetProgram::", "GatedDeltaNetValue2Program::")
    method = replace_once(method,
        "WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, kValueChannelsPerWorkgroup)",
        "WGSL_TEMPLATE_PARAMETER(value_channels_per_workgroup, 2)")
    source = source[:start] + declaration + "\n\n" + method + source[start:]
    signature = "Status GatedDeltaNet::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    if signature not in source:
        signature = "Status GatedDeltaNet::ComputeInternal(ComputeContext& context) const {"
    source = replace_once(source, signature, signature + f'''
  const int value2_mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(value2_mode == 0 || value2_mode == 1, "{FLAG} must be 0 or 1");
''')
    start = source.index("  GatedDeltaNetProgram program{", source.index("const bool vectorized_value_io ="))
    end = source.index(";", start) + 1
    declaration = source[start:end]
    stop = source.index("  return context.RunProgram(program);", end) + len("  return context.RunProgram(program);")
    setup = source[end:stop]
    setup = replace_once(setup, "* value_tiles)", "* dispatch_value_tiles)")
    setup = replace_once(setup, "workgroup_size, kValueChannelsPerWorkgroup)",
                         "workgroup_size, value2 ? 2 : kValueChannelsPerWorkgroup)")
    private = declaration.replace("GatedDeltaNetProgram", "GatedDeltaNetValue2Program")
    replacement = '''
  const bool value2 = value2_mode == 1 && total_tokens == 1 && batch == 1 &&
      use_packed_qkv && dk == 128 && dv == 128 &&
      query->IsDataType<MLFloat16>() &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      static_cast<uint64_t>(hv) * ((dv + 1) / 2) <=
          context.DeviceLimits().maxComputeWorkgroupsPerDimension;
  auto run_program = [&](auto& program, uint64_t dispatch_value_tiles) -> Status {
''' + setup + '''
  };
  if (value2) {
''' + private + '''
    return run_program(program, onnxruntime::narrow<uint32_t>((dv + 1) / 2));
  }
''' + declaration + '''
  return run_program(program, value_tiles);'''
    return source[:start] + replacement + source[stop:]


def transform_compare(source):
    source = source.replace("ORT_TEST_TOGGLE_GDN_VIEW", "ORT_TEST_TOGGLE_GDN_VALUE2")
    source = source.replace("ORT_WEBGPU_GDN_PACKED_DECODE_VIEW", FLAG)
    source = replace_once(source, "run < 3", "run < 9")
    source = replace_once(source, '<< " runs=3"', '<< " runs=9"')
    source = replace_once(source, '    Ort::SessionOptions options;', '''
    const char* pending = std::getenv("ORT_TEST_MAX_PENDING_DISPATCHES");
    if (pending == nullptr || (std::strcmp(pending, "1") != 0 && std::strcmp(pending, "64") != 0))
      throw std::runtime_error("Pending must be 1 or 64");
    const char* capture = std::getenv("ORT_TEST_ENABLE_GRAPH_CAPTURE");
    if (capture == nullptr) capture = "0";
    if (std::strcmp(capture, "0") != 0 && std::strcmp(capture, "1") != 0)
      throw std::runtime_error("Capture must be 0 or 1");
    Ort::SessionOptions options;''')
    source = replace_once(source, '{"enableGraphCapture", "0"}', '{"enableGraphCapture", capture}')
    source = replace_once(source, '{"maxNumPendingDispatches", "64"}', '{"maxNumPendingDispatches", pending}')
    return replace_once(source, '      std::cout << "PASS exact finite outputs "', f'''
      if (std::strcmp(capture, "0") == 0) {{
      set_mid("2");
      bool rejected = false;
      try {{
        fused.Run(Ort::RunOptions{{}}, nullptr, nullptr, 0, names.data(), names.size());
      }} catch (const Ort::Exception& error) {{
        rejected = std::string(error.what()).find("{FLAG} must be 0 or 1") != std::string::npos;
      }}
      if (!rejected) throw std::runtime_error("Invalid GDN value2 selector was not rejected");
      set_mid("0");
      std::cout << "PASS invalid_selector " << path.stem().string() << std::endl;
      }}
      std::cout << "PASS exact finite outputs "''')


def main(*, parent_dir=PARENT, output=OUTPUT, source_transform=transform,
         comparator_transform=transform_compare):
    parent = json.loads((parent_dir / "build-manifest.json").read_text())
    component = json.loads((COMPONENT / "build-manifest.json").read_text())
    original = COMPONENT / "gated_delta_net.cc"
    assert sha(original) == component["source_sha256"][str(original)]
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Combined parent relink mismatch"
    library.unlink()
    command = component["compile_command"].copy()
    original_object = COMPONENT / "gated_delta_net.cc.o"
    reproduced = output / "original-gated_delta_net.cc.o"
    command[command.index("-o") + 1] = str(reproduced)
    subprocess.run(command, cwd=component["working_directory"], env=environment, check=True)
    assert sha(reproduced) == sha(original_object), "Original GDN object reproduction mismatch"
    source = output / original.name
    source.write_text(source_transform(original.read_text(), HEADER.read_text()))
    obj = output / "gated_delta_net.cc.o"
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=component["working_directory"], env=environment, check=True)
    assert link.count(str(original_object)) == 1
    link[link.index(str(original_object))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    comparator = output / "native_compare.cpp"
    comparator.write_text(comparator_transform((COMPONENT / "native_compare.cpp").read_text()))
    subprocess.run(["c++", "-O3", "-std=c++20", str(comparator),
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-lonnxruntime", "-o", str(output / "native_compare")], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        original_gdn_object_byte_identical=True, original_gdn_object_sha256=sha(reproduced),
        original_header_sha256=sha(HEADER), no_existing_kernel_object_layout_changes=True,
        no_per_channel_arithmetic_or_reduction_changes=True, packed_fp16_decode_only=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        comparator_sha256=sha(output / "native_compare"),
        source_sha256={p.name: sha(p) for p in (source, comparator)},
        quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "provider_sha256")}), flush=True)


if __name__ == "__main__":
    main()
