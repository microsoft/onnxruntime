#!/usr/bin/python3
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, help="input symbol file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--output_source", required=True, help="output file")
    parser.add_argument("--version_file", required=True, help="VERSION_NUMBER file")
    parser.add_argument("--style", required=True, choices=["gcc", "vc", "xcode"])
    parser.add_argument("--config", required=True, nargs="+")
    return parser.parse_args()


def read_symbols(filename: str, allow_dups: bool = False):
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue

            if line in symbols:
                if not allow_dups:
                    print("dup symbol: %s", line)
                    exit(-1)
            else:
                symbols.add(line)


args = parse_arguments()
print("Generating symbol file for %s" % str(args.config))
with open(args.version_file, 'r') as f:
    VERSION_STRING = f.read().strip()

print("VERSION:%s" % VERSION_STRING)

symbols = set()
for c in args.config:
    file_name = os.path.join(args.src_root, 'core', 'providers', c, 'symbols.txt')
    with open(file_name, 'r') as file:
        read_symbols(file_name)

# add stubs from session to provide graceful error message if EP isn't included in the build.
# this will have dups for EPs that are enabled in the build which is fine as we #ifdef out the stubs
# in the ORT C API implementation if that is the case
read_symbols(os.path.join(args.src_root, 'core', 'session', 'symbols.txt'), allow_dups=True)

symbols = sorted(symbols)

symbol_index = 1
with open(args.output, 'w') as file:
    if args.style == 'vc':
        file.write('LIBRARY "onnxruntime.dll"\n')
        file.write('EXPORTS\n')
    elif args.style == 'xcode':
        pass    # xcode compile don't has any header.
    else:
        file.write('VERS_%s {\n' % VERSION_STRING)
        file.write(' global:\n')

    for symbol in symbols:
        if args.style == 'vc':
            file.write(" %s @%d\n" % (symbol, symbol_index))
        elif args.style == 'xcode':
            file.write("_%s\n" % symbol)
        else:
            file.write("  %s;\n" % symbol)
        symbol_index += 1

    if args.style == 'gcc':
        file.write(" local:\n")
        file.write("    *;\n")
        file.write("};   \n")

with open(args.output_source, 'w') as file:
    file.write('#include "core/session/onnxruntime_c_api.h"\n')
    for c in args.config:
        # WinML adapter should not be exported in platforms other than Windows.
        # Exporting OrtGetWinMLAdapter is exported without issues using .def file when compiling for Windows
        # so it isn't necessary to include it in generated_source.c
        if c != "winml" and c != "cuda":
            file.write('#include "core/providers/%s/%s_provider_factory.h"\n' % (c, c))
    file.write("void* GetFunctionEntryByName(const char* name){\n")
    for symbol in symbols:
        if symbol != "OrtGetWinMLAdapter":
            file.write("if(strcmp(name,\"%s\") ==0) return (void*)&%s;\n" % (symbol, symbol))
    file.write("return NULL;\n")
    file.write("}\n")
