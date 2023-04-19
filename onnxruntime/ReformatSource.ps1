# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Simply finds all .h and .cc files from the current directory and below, and runs clang-format in-place on them.
# Assumes clang-format is in the path, which means you will need to install clang using a Windows snapshot build from https://llvm.org/builds/
# Requires a .clang-format config file to be in the current directory or a parent directory from where the script is run.
# Expected usage is to run it from its current location so that source in 'core' and 'test' is updated.

gci -Recurse -Include  *.h, *.cc | foreach {
    Write-Host "Updating " $_.FullName
    clang-format -i $_
}
