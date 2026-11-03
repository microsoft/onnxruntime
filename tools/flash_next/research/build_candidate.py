import argparse
from concurrent.futures import ThreadPoolExecutor
import shlex
import subprocess
from pathlib import Path

baseline = Path("/workspace/onnxruntime-webgpu-fp8-checkpoint")
candidate = Path("/workspace/onnxruntime-webgpu-h200-perf")
build = baseline / "build-webgpu-fp8/Release"
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="build-experiment")
parser.add_argument("--syntax-tests", action="store_true")
parser.add_argument("--extra-source", action="append", default=[])
parser.add_argument("--rebuild-header", action="append", default=[])
parser.add_argument("--jobs", type=int, choices=range(1, 5), default=1)
parser.add_argument("--test-source", action="append",
                    default=["test/providers/webgpu/contrib_ops/moe_test.cc"])
args = parser.parse_args()
check_tests = args.syntax_tests
test_sources = args.test_source
out = candidate / args.output
out.mkdir(exist_ok=True)
subprocess.run([
    "python3", str(candidate / "tools/python/wgsl_gen.py"),
    "--output", str(out / "wgsl_template_gen"), "-I", "wgsl_template_gen/",
    "--preserve-code-ref", "--generator", "static-cpp",
    "-i", str(candidate / "onnxruntime/core/providers/webgpu"),
    "-i", str(candidate / "onnxruntime/contrib_ops/webgpu"),
], check=True)
target = build / "CMakeFiles/onnxruntime_providers_webgpu.dir"
flags = {}
for line in (target / "flags.make").read_text().splitlines():
    if " = " in line:
        key, value = line.split(" = ", 1)
        for directory in ("onnxruntime", "include"):
            value = value.replace(str(baseline / directory), str(candidate / directory))
        flags[key] = shlex.split(value)
sources = [
    "contrib_ops/webgpu/moe/moe.cc",
    "contrib_ops/webgpu/moe/qmoe.cc",
    "core/providers/webgpu/wgsl_templates/wgsl_gen.cc",
    "core/providers/webgpu/ep/ep.cc",
    "core/providers/webgpu/webgpu_profiler.cc",
    "core/providers/webgpu/webgpu_execution_provider.cc",
]
sources.extend(args.extra_source)
for header in args.rebuild_header:
    matched = 0
    for dependency in target.rglob("*.o.d"):
        contents = dependency.read_text()
        if str(baseline / "onnxruntime" / header) not in contents:
            continue
        for token in contents.replace("\\\n", "").split():
            source = Path(token)
            if source.suffix == ".cc" and source.is_relative_to(baseline / "onnxruntime"):
                sources.append(str(source.relative_to(baseline / "onnxruntime")))
                matched += 1
                break
        else:
            raise RuntimeError(f"Cannot identify dependent source in {dependency}")
    if not matched:
        raise RuntimeError(f"No compiler dependencies found for changed header {header}")
    print(f"Recompiling {matched} translation units affected by {header}", flush=True)
sources = list(dict.fromkeys(sources))
replacements = {}
def compile_source(source):
    obj = out / (source.replace("/", "__") + ".o")
    subprocess.run(
        [
            "/usr/bin/c++",
            "-I" + str(out),
            *flags["CXX_DEFINES"],
            *flags["CXX_INCLUDES"],
            *flags["CXX_FLAGS"],
            "-include",
            str(candidate / "include/onnxruntime/ep/adapters.h"),
            "-c",
            str(candidate / "onnxruntime" / source),
            "-o",
            str(obj),
        ],
        cwd=build,
        check=True,
    )
    return source + ".o", str(obj)
with ThreadPoolExecutor(max_workers=args.jobs) as executor:
    replacements.update(executor.map(compile_source, sources))
args = []
for arg in shlex.split((target / "link.txt").read_text()):
    args.extend(shlex.split((build / arg[1:]).read_text()) if arg.startswith("@") else [arg])
for index, arg in enumerate(args):
    for suffix, obj in replacements.items():
        if arg.endswith(suffix):
            args[index] = obj
    if index > 0 and args[index - 1] == "-o":
        args[index] = str(out / "libonnxruntime_providers_webgpu.so")
subprocess.run([*args, "-Wl,--no-undefined"], cwd=build, check=True)
print("Built isolated candidate plugin from changed objects; baseline build unchanged.")
if check_tests:
    test_flags = {}
    for line in (build / "CMakeFiles/onnxruntime_test_all.dir/flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            for directory in ("onnxruntime", "include"):
                value = value.replace(str(baseline / directory), str(candidate / directory))
            test_flags[key] = shlex.split(value)
    for source in test_sources:
        subprocess.run([
            "/usr/bin/c++",
            *[flag for flag in test_flags["CXX_DEFINES"] if not flag.startswith("-DUSE_WEBGPU")],
            "-DUSE_WEBGPU=1", *test_flags["CXX_INCLUDES"], *flags["CXX_INCLUDES"],
            *test_flags["CXX_FLAGS"], "-fsyntax-only",
            str(candidate / "onnxruntime" / source),
        ], cwd=build, check=True)
        print(f"Complete regression test translation unit passed syntax validation: {source}")
