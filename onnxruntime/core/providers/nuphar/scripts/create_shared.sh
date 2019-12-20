#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -x -e -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

function usage {
    echo Usage: create_shared.sh -c cache_dir -m input_model_file -o output_so_file
    echo The generated file would be cache_dir/output_so_file
    exit 1
}

while getopts c:m:o: parameter_Option
do case "${parameter_Option}"
in
c) CACHE_DIR=${OPTARG};;
m) MODEL_FILE=${OPTARG};;
o) OUTPUT_SO_FILE=${OPTARG};;
esac
done

if [ -z "$CACHE_DIR" ]; then
    echo "No cache_dir specified"
    usage
fi

if [ -z "$OUTPUT_SO_FILE" ]; then
    OUTPUT_SO_FILE=jit.so
fi

# check required tools
if ! [ -x "$(command -v g++)" ]; then
    echo "Could not find g++"
    exit 1
fi

declare -a all_cc_files

cd $CACHE_DIR
if [ -x "$MODEL_FILE" ]; then
    # generate checksum.cc
    md5=`md5sum ${MODEL_FILE} | awk '{ print $1 }'`
    cat > $CACHE_DIR/checksum.cc <<__EOF__
#include <stdlib.h>
static const char model_checksum[] = "$md5";
extern "C"
void _ORTInternal_GetCheckSum(const char*& cs, size_t& len) {
  cs = model_checksum; len = sizeof(model_checksum)/sizeof(model_checksum[0]) - 1;
}    
__EOF__
    all_cc_files+=(checksum)
fi

# generate cache_version.cc
VERSION_FILE="${SCRIPT_DIR}/NUPHAR_CACHE_VERSION"
cat > $CACHE_DIR/cache_version.cc <<__EOF__
#include "$VERSION_FILE"
extern "C"
const char* _ORTInternal_GetCacheVersion() {
  return __NUPHAR_CACHE_VERSION__;
}
__EOF__
all_cc_files+=(cache_version)

for cc_file in "${all_cc_files[@]}"; do
  g++ -std=c++14 -fPIC -o "$cc_file".o -c "$cc_file".cc
  rm "$cc_file".cc
done

# link
if ls *.o 1> /dev/null 2>&1; then
    OBJS=""
    for o_file in *.o; do
        OBJS+=" $o_file"
    done
        
    if ! [ -z "$OBJS" ]; then
        g++ -shared -fPIC -o $CACHE_DIR/$OUTPUT_SO_FILE $OBJS
    fi
    rm *.o
fi
