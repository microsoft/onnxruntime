import argparse
import os
import shutil

build_dir = ["wasm_SIMD","wasm_threaded","wasm_SIMD_threaded"]
build_flags = [" --enable_wasm_simd"," --enable_wasm_threads"," --enable_wasm_simd --enable_wasm_threads "]

firsttime_flags = '--build_wasm --skip_tests --emsdk_version releases-upstream-2ddc66235392b37e5b33477fd86cbe01a14b8aa2-64bit --cmake_generator "Visual Studio 16 2019"'
default_flags = '--build_wasm --skip_tests --skip_submodule_sync --emsdk_version releases-upstream-2ddc66235392b37e5b33477fd86cbe01a14b8aa2-64bit --cmake_generator "Visual Studio 16 2019"'
BinariesDirectory = '.\\build\\'
Dist_Path = ".\\js\\web\\dist\\"
Binding_Path = ".\\js\\web\\lib\\wasm\\binding\\"

# Needed Arguments
# -b - build WASM
#      - Need to add partial build for only WASM artifacts - reduces the amount of time to build ~100x
#      - Need to add full build for the entire WASM + Web App
#      - configuration - Release/Debug

### generating the build folders
if not os.path.isdir(BinariesDirectory):
    os.mkdir(BinariesDirectory)

for path in build_dir:
    dirctory_name = BinariesDirectory+path
    if not os.path.isdir(dirctory_name):
        os.mkdir(dirctory_name)

if not os.path.isdir(Dist_Path):
    os.mkdir(Dist_Path)

if not os.path.isdir(Binding_Path):
    os.mkdir(Binding_Path)

#### running the WASM build commands

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--build", choices=["Release","Debug"], help='build WASM - {Release,Debug}')

args = parser.parse_args()

if(args.build):
    configuration = args.build
    command = "build.bat "+firsttime_flags+" --config "+configuration+" --build_dir "+BinariesDirectory+"\\wasm"
    print(command)
    ##os.system(command)

    for i,path in enumerate(build_dir):
        command = "build.bat "+default_flags+" --config "+configuration+" --build_dir "+BinariesDirectory+path+" --path_to_protoc_exe "+BinariesDirectory+"\\wasm\\host_protoc\\"+configuration+"\\protoc.exe"+build_flags[i]
        print(command)
        ###os.system(command)

# copy the files to the right location

####

# build NPM
