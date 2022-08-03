import argparse
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname

build_script = "build.sh"
if "win" in sys.platform:
    build_script = "build.bat"

build_dir = ["wasm", "wasm_SIMD", "wasm_threaded", "wasm_SIMD_threaded"]
build_flags = ["", " --enable_wasm_simd", " --enable_wasm_threads", " --enable_wasm_simd --enable_wasm_threads "]
wasm_file_names = ["ort-wasm.wasm", "ort-wasm-simd.wasm", "ort-wasm-threaded.wasm", "ort-wasm-simd-threaded.wasm"]
js_file_names = ["ort-wasm.js", "ort-wasm-threaded.worker.js", "ort-wasm-threaded.js"]
build_dir_js = ["wasm", "wasm_threaded", "wasm_threaded"]

default_flags = " --build_wasm --skip_tests --skip_submodule_sync --parallel "
BinariesDirectory = ".\\build\\"
Dist_Path = ".\\js\\web\\dist\\"
Binding_Path = ".\\js\\web\\lib\\wasm\\binding\\"

# Stage #1 - Creating the folders for the build process
if not os.path.isdir(BinariesDirectory):
    os.mkdir(BinariesDirectory)

for path in build_dir:
    dirctory_name = BinariesDirectory + path
    if not os.path.isdir(dirctory_name):
        os.mkdir(dirctory_name)

if not os.path.isdir(Dist_Path):
    os.mkdir(Dist_Path)

if not os.path.isdir(Binding_Path):
    os.mkdir(Binding_Path)

# Stage #1.1 - reading the script parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    choices=["Release", "Debug", "RelWithDebInfo"],
    help="build WASM artifactsfor the configuration - {Release,Debug,RelWithDebInfo}",
)
parser.add_argument(
    "--copy",
    choices=["Release", "Debug", "RelWithDebInfo"],
    help="copy WASM artifacts to destination folders - {Release,Debug,RelWithDebInfo}",
)
args = parser.parse_args()
configuration = "none"

# Stage #2 - Building the WASM artifacts in 4 different folders listed in - build_dir
if args.config:
    configuration = args.config

    for i, path in enumerate(build_dir):
        command = (
            build_script
            + " --config "
            + configuration
            + default_flags
            + " --build_dir "
            + os.path.join(BinariesDirectory, path)
            + build_flags[i]
        )
        p = subprocess.Popen(command, shell=True)
        p.wait()

# Stage 3 - copy the files to the right location as instructed in the build process for ORT Web
if args.copy:
    configuration = args.copy
    ## Copying WASM artifacts
    for i, file_name in enumerate(wasm_file_names):
        shutil.copyfile(
            os.path.join(BinariesDirectory, build_dir[i], configuration, file_name),
            os.path.join(Dist_Path, file_name),
        )

    ## Copying JS binding files
    for i, file_name in enumerate(js_file_names):
        shutil.copyfile(
            os.path.join(BinariesDirectory, build_dir_js[i], configuration, file_name),
            os.path.join(Binding_Path, file_name),
        )

# Stage 4 - running the npm build
path = abspath(dirname(__file__))
p = subprocess.Popen("npm ci", shell=True, cwd=path)
p.wait()
p = subprocess.Popen("npm ci", shell=True, cwd=path + "\\common\\")
p.wait()
p = subprocess.Popen("npm ci", shell=True, cwd=path + "\\web")
p.wait()
p = subprocess.Popen("npm run build", shell=True, cwd=path + "\\web")
p.wait()
