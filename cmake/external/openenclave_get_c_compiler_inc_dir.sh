#!/usr/bin/env bash
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# obtain default compiler include holding the intrinsics
#
# Args: <CC> - The C-compiler to use
#
function exit_handler()
{
    test "$?" == 0 && return
    echo "An error occured" >&2
    exit 1
}
trap exit_handler EXIT
trap exit ERR

CC=$1

# The PREPROCESS_OUTPUT_FILE is a clang compiler workaround due to a bug that
# exists where piping the output of clang to gawk causes a 'broken pipe' error
# to occur. This seems to happen only when multiple invocations of clang are
# occuring on the same system.
PREPROCESS_OUTPUT_FILE=$(mktemp preprocess-output.XXXX)
echo "#include <x86intrin.h>" | $CC -E - -M > "$PREPROCESS_OUTPUT_FILE"
file=$(gawk '/x86intrin\.h/{$0=gensub("-.o: ","","g"); print $1; exit}' < "$PREPROCESS_OUTPUT_FILE")
rm "$PREPROCESS_OUTPUT_FILE"
echo "$file" | grep -q 'x86intrin\.h' && test -f "$file"
dir=$(dirname "$file")
test -d "$dir"
echo -n "$dir"
