#!/bin/bash

# formats Objective-C/C++ code

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

clang-format -i $(find ${SCRIPT_DIR} -name "*.h" -o -name "*.m" -o -name "*.mm")

