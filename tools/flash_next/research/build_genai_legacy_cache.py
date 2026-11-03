from pathlib import Path
import shlex
import subprocess


source = Path("/workspace/onnxruntime-genai-webgpu-fp8-checkpoint")
build = source / "build-webgpu-fp8/Release"
out = source / "build-h200-int4-legacy"
out.mkdir(exist_ok=True)
flags = {}
for line in (build / "CMakeFiles/onnxruntime-genai-obj.dir/flags.make").read_text().splitlines():
    if " = " in line:
        key, value = line.split(" = ", 1)
        flags[key] = shlex.split(value)
obj = out / "indexer_cache.cpp.o"
subprocess.run(["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"], *flags["CXX_FLAGS"],
                "-c", str(source / "src/models/io/indexer_cache.cpp"), "-o", str(obj)],
               cwd=build, check=True)
command = shlex.split((build / "CMakeFiles/onnxruntime-genai.dir/link.txt").read_text())
for index, value in enumerate(command):
    if value.endswith("/src/models/io/indexer_cache.cpp.o"):
        command[index] = str(obj)
    if index and command[index - 1] == "-o":
        command[index] = str(out / "libonnxruntime-genai.so")
subprocess.run([*command, "-Wl,--no-undefined"], cwd=build, check=True)
print(out / "libonnxruntime-genai.so")
