# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path

$Config = "RelWithDebInfo"
$TimeoutSec = 60 * 60

$CTestExe = (Join-Path $RootDir "ctest.exe")
$CTestTestFile = (Join-Path $RootDir "CTestTestfile.cmake")

# Extract the build path from CTestTestfile.cmake
$OldBuildDirectoryRegex = (Select-String -Path $CTestTestFile -Pattern "# Build directory: (.*)$").Matches.Groups[1].Value
$OldBuildDirectoryBackslashesRegex = ($OldBuildDirectoryRegex -replace "/", "\\\\")

# Substitutions to point CTest at this directory
$NewBuildDirectory = ($RootDir -replace "\\", "/")
$NewBuildDirectoryBackslashes = ($NewBuildDirectory -replace "/", "\\")

# Rewrite CTestTestfile.cmake
(Get-Content $CTestTestFile) `
    -replace $OldBuildDirectoryRegex, $NewBuildDirectory `
    -replace $OldBuildDirectoryBackslashesRegex, $NewBuildDirectoryBackslashes |
    Out-File -Encoding ascii $CTestTestFile

# Run CTest
Push-Location $RootDir
& $CTestExe --build-config $Config --verbose --timeout $TimeoutSec

if (-not $?) {
    throw "Tests failed"
}
