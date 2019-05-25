import argparse
import tvm
import os
import sys
import hashlib

from tvm import contrib
from os import listdir
from os.path import isfile

def parse_arguments():
  parser = argparse.ArgumentParser(description="Offline shared lib creation tool.",
                                   usage='''
Run this under Nuphar cache folder.
The input directory contains .o files and .cc files to link into so name.
                                     ''')
  # Main arguments
  parser.add_argument('--keep_input', action='store_true', help="Keep input files after created so.")
  parser.add_argument('--output_name', help="The output so file name.", default='jit.so')
  parser.add_argument('--input_model', help="The input model file name to generate checksum into shared lib.", default=None)
  return parser.parse_args()

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

def compile_cc(cc_file, o_file):
  if is_windows():
    os.system("cl /Fo" + o_file + " /c " + cc_file)
  else:
    os.system("g++ -std=c++14 -fPIC -o " + o_file + ".o -c " + cc_file)
  
def gen_checksum(file_checksum, keep_input):
  if not file_checksum:
    return

  name = 'ORTInternal_checksum'
  with open(name + '.cc', 'w') as checksum_cc:
    print("#include <stdlib.h>", file=checksum_cc)
    print("static const char model_checksum[] = \"" + file_checksum + "\";", file=checksum_cc)
    print("extern \"C\"", file=checksum_cc)
    if is_windows():
      print("__declspec(dllexport)", file=checksum_cc)
    print("void _ORTInternal_GetCheckSum(const char*& cs, size_t& len) {", file=checksum_cc)
    print("  cs = model_checksum; len = sizeof(model_checksum)/sizeof(model_checksum[0]) - 1;", file=checksum_cc)
    print("}", file=checksum_cc)

  compile_cc(name + '.cc', name + '.o')

  if not keep_input:
    os.remove(name + '.cc')

if __name__ == '__main__':
  args = parse_arguments()
  if args.input_model:
    input_checksum = gen_md5(args.input_model)
    gen_checksum(input_checksum, args.keep_input)

  objs = [f for f in listdir('.') if isfile(f) and '.o' == os.path.splitext(f)[1]]
  contrib.cc.create_shared(args.output_name, objs)
  if not args.keep_input:
    for f in objs:
      os.remove(f)
