import json

from build_qmoe_node_replay import replace_once, sha
from prepare_dense_long_k_native_v2 import ROOT


def native_source(source):
    source = replace_once(source, "    const auto private_value = []() {",
                          '    const char* profiling_env = std::getenv("ORT_TEST_DENSE_NATIVE_PROFILE");\n'
                          '    const std::string profiling = profiling_env == nullptr ? "" : profiling_env;\n'
                          '    if (profiling != "0" && profiling != "1")\n'
                          '      throw std::runtime_error("Explicit native profiling selector must be 0 or 1");\n'
                          "    const auto private_value = []() {")
    source = replace_once(source, "      options.EnableProfiling(prefix.c_str());",
                          '      if (profiling == "1") options.EnableProfiling(prefix.c_str());')
    source = replace_once(source,
                          "      auto profile = session.EndProfilingAllocated(allocator);\n"
                          '      std::cout << "PROFILE " << profile.get() << \'\\n\';',
                          '      if (profiling == "1") {\n'
                          "        auto profile = session.EndProfilingAllocated(allocator);\n"
                          '        std::cout << "PROFILE " << profile.get() << \'\\n\';\n'
                          "      }")
    return replace_once(source, '<< " fp32=" << fp32 << std::endl;',
                        '<< " fp32=" << fp32 << " profiling=" << profiling << std::endl;')


def controller_source(source):
    source = source.replace("native-full-v2", "native-full-v3").replace("native-proof-v2", "native-proof-v3").replace(
        "dense-tile-parallel3-long-k-native-v2", "dense-tile-parallel3-long-k-native-v3")
    source = replace_once(source, "def verify_replay(rows, fp32, requests):\n    if fp32:",
                          "def verify_replay(rows, fp32, requests, profiling):\n    if fp32 or profiling:")
    source = replace_once(source, "if len(rows) != 27 or any(", "if len(rows) != 9 or any(")
    source = replace_once(source, "def command_for(provider, mode, directory, prefix, fixtures, selected, fp32):",
                          "def command_for(provider, mode, directory, prefix, fixtures, selected, fp32, profiling=1):")
    source = replace_once(source, 'if selected not in (0, 1, 2) or fp32 not in (0, 1) or mode not in ("capture", "compare"):',
                          'if (selected not in (0, 1, 2) or fp32 not in (0, 1) or profiling not in (0, 1)\n'
                          '            or mode not in ("capture", "compare")):')
    source = replace_once(source, 'f"export {FLAG}={selected} ORT_TEST_DENSE_FP32_ACCUMULATION={fp32} "',
                          'f"export {FLAG}={selected} ORT_TEST_DENSE_FP32_ACCUMULATION={fp32} "\n'
                          '        f"ORT_TEST_DENSE_NATIVE_PROFILE={profiling} "')
    source = replace_once(
        source,
        '    for phase, library, mode, selected, fp32 in (\n'
        '            ("original", original, "capture", 0, 0), ("default-off", provider, "compare", 0, 0),\n'
        '            ("selected", provider, "compare", 1, 0), ("original-after", original, "compare", 0, 0),\n'
        '            ("fp32-original", original, "capture", 0, 1), ("fp32-selected", provider, "compare", 1, 1)):',
        '    for phase, library, mode, selected, fp32, profiling in (\n'
        '            ("original", original, "capture", 0, 0, 1), ("default-off", provider, "compare", 0, 0, 1),\n'
        '            ("selected", provider, "compare", 1, 0, 1), ("original-after", original, "compare", 0, 0, 1),\n'
        '            ("fp32-original", original, "capture", 0, 1, 1), ("fp32-selected", provider, "compare", 1, 1, 1),\n'
        '            ("replay-original", original, "compare", 0, 0, 0),\n'
        '            ("replay-default-off", provider, "compare", 0, 0, 0),\n'
        '            ("replay-selected", provider, "compare", 1, 0, 0),\n'
        '            ("replay-original-after", original, "compare", 0, 0, 0),\n'
        '            ("replay-fp32-original", original, "compare", 0, 1, 0),\n'
        '            ("replay-fp32-selected", provider, "compare", 1, 1, 0)):')
    source = replace_once(source, "command_for(library, mode, directory, prefix, fixtures, selected, fp32)",
                          "command_for(library, mode, directory, prefix, fixtures, selected, fp32, profiling)")
    source = replace_once(source, "pending=(1|64) fp32=(0|1)$", "pending=(1|64) fp32=(0|1) profiling=(0|1)$")
    source = replace_once(source, "len(profiles) != 27", "len(profiles) != (27 if profiling else 0)")
    source = replace_once(source, "zip(rows, profiles, fixtures, strict=True)",
                          "zip(rows, profiles if profiling else [None] * 27, fixtures, strict=True)")
    source = replace_once(source, 'or row[3] != ("1" if index % 2 == 0 else "64") or int(row[4]) != fp32):',
                          'or row[3] != ("1" if index % 2 == 0 else "64") or int(row[4]) != fp32\n'
                          "                    or int(row[5]) != profiling):")
    source = replace_once(
        source,
        "            profile = host_path(profile_name)\n"
        "            verify_profile(json.loads(profile.read_text()),\n"
        "                           bool(selected and not fp32 and (k, n, count_rows) == (6144, 2560, 1)),\n"
        '                           256 if mode == "capture" else 128)',
        "            if profile_name is not None:\n"
        "                profile = host_path(profile_name)\n"
        "                verify_profile(json.loads(profile.read_text()),\n"
        "                               bool(selected and not fp32 and (k, n, count_rows) == (6144, 2560, 1)),\n"
        '                               256 if mode == "capture" else 128)\n'
        "                evidence.append(profile)")
    source = replace_once(source, "evidence.extend([profile, reference])", "evidence.append(reference)")
    source = replace_once(source, "if fp32 and not stats.exists():",
                          "if (fp32 or profiling) and not stats.exists():")
    source = replace_once(source, 'verify_replay(replay, fp32, 256 if mode == "capture" else 128)',
                          'verify_replay(replay, fp32, 256 if mode == "capture" else 128, profiling)')
    return replace_once(source, '"fp32_accumulation_fallback": bool(fp32), "replay_instances": len(replay),',
                        '"fp32_accumulation_fallback": bool(fp32), "replay_instances": len(replay),\n'
                        '            "profiling_enabled": bool(profiling),\n'
                        '            "expected_single_row_replay_instances": 9 if not fp32 and not profiling else 0,')


def main():
    parents = (ROOT / "dense_long_k_fixed_native_v2.cpp", ROOT / "build_dense_long_k_native_v2.py",
               ROOT / "run_dense_long_k_native_v2.py")
    outputs = (ROOT / "dense_long_k_fixed_native_v3.cpp", ROOT / "build_dense_long_k_native_v3.py",
               ROOT / "run_dense_long_k_native_v3.py")
    report = ROOT / "dense-long-k-native-v3-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh split profile/replay native generation required")
    for parent, output, transform in zip(parents, outputs, (
            native_source,
            lambda text: text.replace("native-full-v2", "native-full-v3").replace(
                "dense_long_k_fixed_native_v2.cpp", "dense_long_k_fixed_native_v3.cpp"),
            controller_source), strict=True):
        text = transform(parent.read_text())
        if output.suffix == ".py":
            compile(text, str(output), "exec")
        with output.open("x") as stream:
            stream.write(text)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "reason": "Sealed replay bypasses profiled calls and rows>1. Separately prove all original goldens with profiling off and nine real single-row replay instances.",
                   "source_sha256": {str(path): sha(path) for path in (ROOT / __file__, *parents, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
