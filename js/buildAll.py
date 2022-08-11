import argparse
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname

Build_Script = "build.sh"
CONST_VALUE_Default_Flags = " --build_wasm --skip_tests --skip_submodule_sync --parallel "
CONST_VALUE_Binaries_Directory = "./build/"
CONST_VALUE_Dist_Path = "./js/web/dist/"
CONST_VALUE_Binding_Path = "./js/web/lib/wasm/binding/"

if sys.platform.startswith("win"):
    Build_Script = "build.bat"
    CONST_VALUE_Binaries_Directory = ".\\build\\"
    CONST_VALUE_Dist_Path = ".\\js\\web\\dist\\"
    CONST_VALUE_Binding_Path = ".\\js\\web\\lib\\wasm\\binding\\"

CONST_VALUE_Builds = [
    {"dir": "wasm", "wasm_flags": "", "wasm_file_name": "ort-wasm.wasm"},
    {"dir": "wasm_SIMD", "wasm_flags": "--enable_wasm_simd", "wasm_file_name": "ort-wasm-simd.wasm"},
    {"dir": "wasm_threaded", "wasm_flags": "--enable_wasm_threads", "wasm_file_name": "ort-wasm-threaded.wasm"},
    {
        "dir": "wasm_SIMD_threaded",
        "wasm_flags": "--enable_wasm_simd --enable_wasm_threads ",
        "wasm_file_name": "ort-wasm-simd-threaded.wasm",
    },
]

CONST_VALUE_JS_file_names = [
    {"dir": "wasm", "file_name": "ort-wasm.js"},
    {"dir": "wasm_threaded", "file_name": "ort-wasm-threaded.worker.js"},
    {"dir": "wasm_threaded", "file_name": "ort-wasm-threaded.js"},
]

CONST_VALUE_Npm_Build_Dir = [
    {"command": "npm ci", "path": abspath(dirname(__file__))},
    {"command": "npm ci", "path": os.path.join(abspath(dirname(__file__)), "common")},
    {"command": "npm ci", "path": os.path.join(abspath(dirname(__file__)), "web")},
    {"command": "npm run build", "path": os.path.join(abspath(dirname(__file__)), "web")},
]

#### generating the build folders
if not os.path.isdir(CONST_VALUE_Binaries_Directory):
    os.mkdir(CONST_VALUE_Binaries_Directory)

for entry in CONST_VALUE_Builds:
    dirctory_name = os.path.join(CONST_VALUE_Binaries_Directory, entry["dir"])
    if not os.path.isdir(dirctory_name):
        os.mkdir(dirctory_name)

if not os.path.isdir(CONST_VALUE_Dist_Path):
    os.mkdir(CONST_VALUE_Dist_Path)

if not os.path.isdir(CONST_VALUE_Binding_Path):
    os.mkdir(CONST_VALUE_Binding_Path)

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

    for entry in CONST_VALUE_Builds:
        command = (
            Build_Script
            + " --config "
            + configuration
            + " "
            + CONST_VALUE_Default_Flags
            + " --build_dir "
            + os.path.join('"' + CONST_VALUE_Binaries_Directory, entry["dir"] + '"')
            + " "
            + entry["wasm_flags"]
        )
        p = subprocess.Popen(command, shell=True)
        p.wait()

#### copy the files to the right location
if args.copy:
    configuration = args.copy
    ## Copying WASM artifacts
    for entry in CONST_VALUE_Builds:
        if not os.path.exists(
            os.path.join(CONST_VALUE_Binaries_Directory, entry["dir"], configuration, entry["wasm_file_name"])
        ):
            print("Error - can find " + entry["wasm_file_name"] + " there might be an issue with the build\n")
            exit()
        shutil.copyfile(
            os.path.join(CONST_VALUE_Binaries_Directory, entry["dir"], configuration, entry["wasm_file_name"]),
            os.path.join(CONST_VALUE_Dist_Path, entry["wasm_file_name"]),
        )

    ## Copying JS binding files
    for entry in CONST_VALUE_JS_file_names:
        if not os.path.exists(
            os.path.join(CONST_VALUE_Binaries_Directory, entry["dir"], configuration, entry["file_name"])
        ):
            print("Error - can find " + entry["file_name"] + " there might be an issue with the build\n")
            exit()
        shutil.copyfile(
            os.path.join(CONST_VALUE_Binaries_Directory, entry["dir"], configuration, entry["file_name"]),
            os.path.join(CONST_VALUE_Binding_Path, entry["file_name"]),
        )

#### build NPM package
for entry in CONST_VALUE_Npm_Build_Dir:
    p = subprocess.Popen(entry["command"], shell=True, cwd=entry["path"])
    p.wait()
