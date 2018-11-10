#!/usr/bin/python3
import sys
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, help="input symbol file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--style", required=True, choices=["gcc", "vc"])
    parser.add_argument("--config", required=True, nargs="+")
    return parser.parse_args()

args = parse_arguments()
print("Generating symbol file for %s" % str(args.config))
symbols = set()
for c in args.config:
  file_name = os.path.join(args.src_root,'core', 'providers',c,'symbols.txt')
  with open(file_name, 'r') as file:
    for line in file:
      line = line.strip()
      if line in symbols:
        print("dup symbol: %s", line)
        exit(-1)
      symbols.add(line)
symbols = sorted(symbols)

symbol_index = 1
with open(args.output, 'w') as file:
  if args.style == 'vc':    
    file.write('LIBRARY "onnxruntime.dll"\n')
    file.write('EXPORTS\n')
  else:
    file.write('VERS_1.0 {\n')
    file.write(' global:\n')

  for symbol in symbols:
    if args.style == 'vc':
      file.write(" %s @%d\n" % (symbol,symbol_index))
    else:
      file.write("  %s;\n" % symbol)
    symbol_index +=1

  if args.style == 'gcc':
    file.write(" local:\n")
    file.write("    *;\n")
    file.write("};   \n")
