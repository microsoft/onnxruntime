import argparse
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname

BUILD_SCRIPT = "source ../build.sh"
DEFAULT_FLAGS = " --build_wasm --skip_tests --skip_submodule_sync --parallel "
BINARIES_DIR = os.path.normpath("../build/")
DIST_PATH = os.path.normpath("./web/dist/")
BINDING_PATH = os.path.normpath("./web/lib/wasm/binding/")

if sys.platform.startswith("win"):
    BUILD_SCRIPT = "call ..\\build.bat"

BUILDS = [
    {"dir": "wasm", "wasm_flags": "", "wasm_file_name": "ort-wasm.wasm"},
    {"dir": "wasm_SIMD", "wasm_flags": "--enable_wasm_simd", "wasm_file_name": "ort-wasm-simd.wasm"},
    {"dir": "wasm_threaded", "wasm_flags": "--enable_wasm_threads", "wasm_file_name": "ort-wasm-threaded.wasm"},
    {
        "dir": "wasm_SIMD_threaded",
        "wasm_flags": "--enable_wasm_simd --enable_wasm_threads ",
        "wasm_file_name": "ort-wasm-simd-threaded.wasm",
    },
]

JS_FILES = [
    {"dir": "wasm", "file_name": "ort-wasm.js"},
    {"dir": "wasm_threaded", "file_name": "ort-wasm-threaded.worker.js"},
    {"dir": "wasm_threaded", "file_name": "ort-wasm-threaded.js"},
]

NPM_BUILD_DIR = [
    {"command": "npm ci", "path": os.path.normpath(os.path.join(abspath(dirname(__file__)), "../"))},
    {"command": "npm ci", "path": os.path.normpath(os.path.join(abspath(dirname(__file__)), "../common"))},
    {"command": "npm ci", "path": os.path.normpath(os.path.join(abspath(dirname(__file__)), "../web"))},
    {"command": "npm run build", "path": os.path.normpath(os.path.join(abspath(dirname(__file__)), "../web"))},
]

#### Generating the build folders
if not os.path.isdir(BINARIES_DIR):
    os.mkdir(BINARIES_DIR)

for entry in BUILDS:
    dirctory_name = os.path.join(BINARIES_DIR, entry["dir"])
    if not os.path.isdir(dirctory_name):
        os.mkdir(dirctory_name)

if not os.path.isdir(DIST_PATH):
    os.mkdir(DIST_PATH)

if not os.path.isdir(BINDING_PATH):
    os.mkdir(BINDING_PATH)

#### Running the WASM build commands
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    choices=["Release", "Debug", "RelWithDebInfo"],
    help="build WASM artifactsfor the configuration - {Release,Debug,RelWithDebInfo}",
)

args = parser.parse_args()
configuration = "none"

if args.config:
    configuration = args.config

    for entry in BUILDS:
        command = (
            BUILD_SCRIPT
            + " --config "
            + configuration
            + " "
            + DEFAULT_FLAGS
            + " --build_dir "
            + os.path.join('"' + BINARIES_DIR, entry["dir"] + '"')
            + " "
            + entry["wasm_flags"]
        )
        p = subprocess.Popen(command, shell=True)
        p.wait()
        if(not os.path.exists(os.path.join(BINARIES_DIR, entry["dir"], configuration, entry["wasm_file_name"]))and (p.returncode!=0)):
            print("Error - can find " + entry["wasm_file_name"] + " there might be an issue with the build\n")
            exit()

## Copying WASM artifacts
for entry in BUILDS:
    if not os.path.exists(os.path.join(BINARIES_DIR, entry["dir"], configuration, entry["wasm_file_name"])):
        print("Error - can find " + entry["wasm_file_name"] + " there might be an issue with the build\n")
        exit()
    shutil.copyfile(
        os.path.join(BINARIES_DIR, entry["dir"], configuration, entry["wasm_file_name"]),
        os.path.join(DIST_PATH, entry["wasm_file_name"]),
    )

## Copying JS binding files
for entry in JS_FILES:
    if not os.path.exists(os.path.join(BINARIES_DIR, entry["dir"], configuration, entry["file_name"])):
        print("Error - can find " + entry["file_name"] + " there might be an issue with the build\n")
        exit()
    shutil.copyfile(
        os.path.join(BINARIES_DIR, entry["dir"], configuration, entry["file_name"]),
        os.path.join(BINDING_PATH, entry["file_name"]),
    )

#### Build NPM package
for entry in NPM_BUILD_DIR:
    p = subprocess.Popen(entry["command"], shell=True, cwd=entry["path"])
    p.wait()
