import argparse
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname, join

build_dir = ["wasm_SIMD","wasm_threaded","wasm_SIMD_threaded"]
build_flags = [" --enable_wasm_simd"," --enable_wasm_threads"," --enable_wasm_simd --enable_wasm_threads "]

firsttime_flags = '--build_wasm --skip_tests --emsdk_version releases-upstream-2ddc66235392b37e5b33477fd86cbe01a14b8aa2-64bit --cmake_generator "Visual Studio 16 2019"'
default_flags = ' --build_wasm --skip_tests --skip_submodule_sync --emsdk_version releases-upstream-2ddc66235392b37e5b33477fd86cbe01a14b8aa2-64bit --cmake_generator "Visual Studio 16 2019"'
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
parser.add_argument("-c", "--copy", choices=["Release","Debug"], help='copy WASM artifacts - {Release,Debug}')
args = parser.parse_args()
configuration = "none"

if(args.build):
    configuration = args.build
    command = "build.bat "+firsttime_flags+" --config "+configuration+" --build_dir "+BinariesDirectory+"\\wasm"
    #print(command)
    os.system(command)

    for i,path in enumerate(build_dir):
        command = "build.bat --config "+configuration+default_flags+" --build_dir "+BinariesDirectory+path+build_flags[i]
        #print(command)
        os.system(command)

# copy the files to the right location
if(args.copy):
    configuration = "\\"+args.copy+"\\"
    ## Copying WASM artifacts
    file_name = "ort-wasm.wasm"
    shutil.copyfile(BinariesDirectory+"\\wasm\\"+configuration+file_name,Dist_Path+file_name)
    file_name = "ort-wasm-simd.wasm"
    shutil.copyfile(BinariesDirectory+build_dir[0]+configuration+file_name,Dist_Path+file_name)
    file_name = "ort-wasm-threaded.wasm"
    shutil.copyfile(BinariesDirectory+build_dir[1]+configuration+file_name,Dist_Path+file_name)
    file_name = "ort-wasm-simd-threaded.wasm"
    shutil.copyfile(BinariesDirectory+build_dir[2]+configuration+file_name,Dist_Path+file_name)

    ## Copying JS binding files
    file_name = "ort-wasm.js"
    shutil.copyfile(BinariesDirectory+"\\wasm\\"+configuration+file_name,Binding_Path+file_name)
    file_name = "ort-wasm-threaded.worker.js"
    shutil.copyfile(BinariesDirectory+build_dir[1]+configuration+file_name,Binding_Path+file_name)
    file_name = "ort-wasm-threaded.js"
    shutil.copyfile(BinariesDirectory+build_dir[1]+configuration+file_name,Binding_Path+file_name)
####
# build NPM
path = abspath(dirname(__file__))
p = subprocess.Popen('npm ci',shell=True,cwd=path)
p.wait()
p = subprocess.Popen('npm ci',shell=True,cwd=path+'\\common\\')
p.wait()
p = subprocess.Popen('npm ci',shell=True,cwd=path+'\\web')
p.wait()
p = subprocess.Popen('npm run build',shell=True,cwd=path+'\\web')
p.wait()
