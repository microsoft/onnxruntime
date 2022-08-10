import argparse
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname

Build_Script = "build.sh"
if sys.platform.startswith("win"):
    Build_Script = "build.bat"

Builds = [
    {"dir": "wasm", "wasm_flags": "", "wasm_file_name": "ort-wasm.wasm"},
    {"dir": "wasm_SIMD", "wasm_flags": "--enable_wasm_simd", "wasm_file_name": "ort-wasm-simd.wasm"},
    {"dir": "wasm_threaded", "wasm_flags": "--enable_wasm_threads", "wasm_file_name": "ort-wasm-threaded.wasm"},
    {
        "dir": "wasm_SIMD_threaded",
        "wasm_flags": "--enable_wasm_simd --enable_wasm_threads ",
        "wasm_file_name": "ort-wasm-simd-threaded.wasm",
    },
]

JS_file_names = [
    {"dir": "wasm", "file_name": "ort-wasm.js"},
    {"dir": "wasm_threaded", "file_name": "ort-wasm-threaded.worker.js"},
    {"dir": "wasm_threaded", "file_name": "ort-wasm-threaded.js"},
]

Npm_Build_Dir = [
    {"command": "npm ci", "path": abspath(dirname(__file__))},
    {"command": "npm ci", "path": os.path.join(abspath(dirname(__file__)), "common")},
    {"command": "npm ci", "path": os.path.join(abspath(dirname(__file__)), "web")},
    {"command": "npm run build", "path": os.path.join(abspath(dirname(__file__)), "web")},
]

Default_Flags = " --build_wasm --skip_tests --skip_submodule_sync --parallel "
Binaries_Directory = ".\\build\\"
Dist_Path = ".\\js\\web\\dist\\"
Binding_Path = ".\\js\\web\\lib\\wasm\\binding\\"

#### generating the build folders
if not os.path.isdir(Binaries_Directory):
    os.mkdir(Binaries_Directory)

for entry in Builds:
    dirctory_name = os.path.join(Binaries_Directory, entry["dir"])
    if not os.path.isdir(dirctory_name):
        os.mkdir(dirctory_name)

if not os.path.isdir(Dist_Path):
    os.mkdir(Dist_Path)

if not os.path.isdir(Binding_Path):
    os.mkdir(Binding_Path)

#### running the WASM build commands
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

if args.config:
    configuration = args.config

    for entry in Builds:
        command = (
            Build_Script
            + " --config "
            + configuration
            + " "
            + Default_Flags
            + " --build_dir "
            + os.path.join('"' + Binaries_Directory, entry["dir"] + '"')
            + " "
            + entry["wasm_flags"]
        )
        p = subprocess.Popen(command, shell=True)
        p.wait()

#### copy the files to the right location
if args.copy:
    configuration = args.copy
    ## Copying WASM artifacts
    for entry in Builds:
        if not os.path.isdir(os.path.join(Binaries_Directory, entry["dir"], configuration, entry["wasm_file_name"])):
            print("Error - can find " + entry["wasm_file_name"] + " there might be an issue with the build\n")
            exit()
        shutil.copyfile(
            os.path.join(Binaries_Directory, entry["dir"], configuration, entry["wasm_file_name"]),
            os.path.join(Dist_Path, entry["wasm_file_name"]),
        )

    ## Copying JS binding files
    for entry in JS_file_names:
        if not os.path.isdir(os.path.join(Binaries_Directory, entry["dir"], configuration, entry["file_name"])):
            print("Error - can find " + entry["file_name"] + " there might be an issue with the build\n")
            exit()
        shutil.copyfile(
            os.path.join(Binaries_Directory, entry["dir"], configuration, entry["file_name"]),
            os.path.join(Binding_Path, entry["file_name"]),
        )

#### build NPM package
for entry in Npm_Build_Dir:
    p = subprocess.Popen(entry["command"], shell=True, cwd=entry["path"])
    p.wait()
