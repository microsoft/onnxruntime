import argparse
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname

build_script = "build.sh"
if 'win' in sys.platform:
    build_script = "build.bat"

builds = [
    {'dir':"wasm", 'wasm_flags':"", 'wasm_file_name':"ort-wasm.wasm"},
    {'dir':"wasm_SIMD", 'wasm_flags':" --enable_wasm_simd", 'wasm_file_name':"ort-wasm-simd.wasm"},
    {'dir':"wasm_threaded", 'wasm_flags':" --enable_wasm_threads", 'wasm_file_name':"ort-wasm-threaded.wasm"},
    {'dir':"wasm_SIMD_threaded", 'wasm_flags':" --enable_wasm_simd --enable_wasm_threads ", 'wasm_file_name':"ort-wasm-simd-threaded.wasm"},
]

js_file_names = [
    {'dir':"wasm",'file_name':"ort-wasm.js"},
    {'dir':"wasm_threaded", 'file_name':"ort-wasm-threaded.worker.js"},
    {'dir':"wasm_threaded", 'file_name':"ort-wasm-threaded.js"}
]

npm_build_dir = [
    abspath(dirname(__file__)),
    os.path.join(abspath(dirname(__file__)),"common"),
    os.path.join(abspath(dirname(__file__)),"web"),
    os.path.join(abspath(dirname(__file__)),"web")
]

default_flags = " --build_wasm --skip_tests --skip_submodule_sync --parallel "
BinariesDirectory = ".\\build\\"
Dist_Path = ".\\js\\web\\dist\\"
Binding_Path = ".\\js\\web\\lib\\wasm\\binding\\"

### generating the build folders
if not os.path.isdir(BinariesDirectory):
    os.mkdir(BinariesDirectory)

for entry in builds:
    dirctory_name = BinariesDirectory + entry['dir']
    if not os.path.isdir(dirctory_name):
        os.mkdir(dirctory_name)

if not os.path.isdir(Dist_Path):
    os.mkdir(Dist_Path)

if not os.path.isdir(Binding_Path):
    os.mkdir(Binding_Path)

#### running the WASM build commands
parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["Release", "Debug","RelWithDebInfo"], help="build WASM artifactsfor the configuration - {Release,Debug,RelWithDebInfo}")
parser.add_argument("--copy", choices=["Release", "Debug","RelWithDebInfo"], help="copy WASM artifacts to destination folders - {Release,Debug,RelWithDebInfo}")
args = parser.parse_args()
configuration = "none"

if args.config:
    configuration = args.config

    for i, entry in enumerate(builds):
        command = (
            build_script
            + " --config "
            + configuration
            + default_flags
            + " --build_dir "
            + os.path.join('"'+BinariesDirectory,builds[i]['dir']+'"')
            + builds[i]['wasm_flags']
        )
        p = subprocess.Popen(command,shell=True)
        p.wait()

# copy the files to the right location
if args.copy:
    configuration = args.copy
    ## Copying WASM artifacts
    for entry in builds:
        file_name = entry['wasm_file_name']
        shutil.copyfile(os.path.join(BinariesDirectory, entry['dir'], configuration, entry['wasm_file_name']), os.path.join(Dist_Path, entry['wasm_file_name']))

    ## Copying JS binding files
    for entry in js_file_names:
        shutil.copyfile(os.path.join(BinariesDirectory, entry['dir'], configuration, entry['file_name']), os.path.join(Binding_Path, entry['file_name']))

####
# build NPM
for path in npm_build_dir:
    p = subprocess.Popen("npm ci", shell=True, cwd=path)
    p.wait()
