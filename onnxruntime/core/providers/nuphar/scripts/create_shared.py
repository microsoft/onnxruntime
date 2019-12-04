# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import argparse
import hashlib
import os
import subprocess
import sys

def is_windows():
    return sys.platform.startswith("win")

def gen_md5(filename):
    if not os.path.exists(filename):
        return False
    hash_md5 = hashlib.md5()
    BLOCKSIZE = 1024*64
    with open(filename, "rb") as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hash_md5.update(buf)
            buf = f.read(BLOCKSIZE)
    return hash_md5.hexdigest()

def gen_checksum(file_checksum, input_dir):
    if not file_checksum:
        return

    name = 'ORTInternal_checksum'
    with open(os.path.join(input_dir, name + '.cc'), 'w') as checksum_cc:
        print('#include <stdlib.h>', file=checksum_cc)
        print('static const char model_checksum[] = "' + file_checksum + '";', file=checksum_cc)
        print('extern "C"', file=checksum_cc)
        if is_windows():
            print('__declspec(dllexport)', file=checksum_cc)
        print('void _ORTInternal_GetCheckSum(const char*& cs, size_t& len) {', file=checksum_cc)
        print('    cs = model_checksum; len = sizeof(model_checksum)/sizeof(model_checksum[0]) - 1;', file=checksum_cc)
        print('}', file=checksum_cc)

def compile_all_cc(path):
    for f in os.listdir(path):
        name, ext = os.path.splitext(f)
        if ext != '.cc':
            continue
        if is_windows():
            subprocess.run(['cl', '/Fo' + name + '.o', '/c', f], cwd=path, check=True)
        else:
            subprocess.run(['g++', '-std=c++14', '-fPIC', '-o', name + '.o', '-c', f], cwd=path, check=True)
        os.remove(os.path.join(path, f))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Offline shared lib creation tool.")
    # Main arguments
    parser.add_argument('--keep_input', action='store_true', help="Keep input files after created so.")
    parser.add_argument('--input_dir', help="The input directory that contains obj files.", required=True)
    parser.add_argument('--output_name', help="The output so file name.", default='jit.so')
    parser.add_argument('--input_model', help="The input model file name to generate checksum into shared lib.", default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.input_model:
        input_checksum = gen_md5(args.input_model)
        gen_checksum(input_checksum, args.input_dir)

    if is_windows():
        # create dllmain
        name = 'ORTInternal_dllmain'
        with open(os.path.join(args.input_dir, name + '.cc'), 'w') as dllmain_cc:
            print("#include <windows.h>", file=dllmain_cc)
            print("BOOL APIENTRY DllMain(HMODULE hModule,", file=dllmain_cc)
            print("                      DWORD   ul_reason_for_call,", file=dllmain_cc)
            print("                      LPVOID  lpReserved)", file=dllmain_cc)
            print(" {return TRUE;}", file=dllmain_cc)

    compile_all_cc(args.input_dir)
    objs = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f)) and '.o' == os.path.splitext(f)[1]]

    if is_windows():
        subprocess.run(['link', '-dll', '-FORCE:MULTIPLE', '-EXPORT:__tvm_main__', '-out:' + args.output_name, '*.o'], cwd=args.input_dir, check=True)
    else:
        subprocess.run(['g++', '-shared', '-fPIC', '-o', args.output_name] + objs, cwd=args.input_dir, check=True)

    if not args.keep_input:
        for f in objs:
            os.remove(os.path.join(args.input_dir, f))