# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script depends on python.exe, cmake.exe and Visual C++ spectre-mitigated libs.
# Please setup AGENT_TEMPDIRECTORY env variable before running this script

 param (
    [string]$cpu_arch = "x64",
    [string]$build_config = "RelWithDebInfo",
    [string]$install_prefix = "."
 )

. "$PSScriptRoot\helpers.ps1"

$ort_src_root = (Get-Item $PSScriptRoot).parent.parent.parent.parent.FullName

Write-Host "ONNX Runtime src root: $ort_src_root"

$ErrorActionPreference = "Stop"

$Env:Path = "$install_prefix\bin;" + $env:Path
$Env:MSBUILDDISABLENODEREUSE=1

New-Item -Path "$install_prefix" -ItemType Directory -Force

# Setup compile flags
$compile_flags = '/MP /guard:cf /Qspectre /DWIN32 /D_WINDOWS /DWINVER=0x0601 /D_WIN32_WINNT=0x0601 /DNTDDI_VERSION=0x06010000 /W3 '
$linker_flags=@('/guard:cf')

if($build_config -eq 'Release'){
  $compile_flags += "/O2", "/Ob2", "/DNDEBUG", "/Gw", "/GL"
} elseif($build_config -eq 'RelWithDebInfo'){
  $compile_flags += "/Zi", "/O2", "/Ob1", "/DNDEBUG", "/Gw", "/GL"
} elseif($build_config -eq 'Debug'){
  $compile_flags += "/Zi", "/Ob0", "/Od", "/RTC1"
} elseif($build_config -eq 'MinSizeRel'){
  $compile_flags += "/O1", "/Ob1", "/DNDEBUG", "/Gw", "/GL"
}
# cmake args that applies to every 3rd-party library
[string[]]$cmake_extra_args="-DCMAKE_CXX_STANDARD=17 `"-DCMAKE_CXX_FLAGS=$compile_flags /EHsc`" ", "`"-DCMAKE_C_FLAGS=$compile_flags`"", "--compile-no-warning-as-error", "--fresh", "-Wno-dev"
if($cpu_arch -eq 'x86'){
  $cmake_extra_args +=  "-A", "Win32", "-T", "host=x64"
  $linker_flags += '/machine:x86'
} elseif($cpu_arch -eq 'x64') {
  $linker_flags += '/machine:x64'
} else {
  throw "$cpu_arch is not supported"
}


$cmake_extra_args += "-DCMAKE_EXE_LINKER_FLAGS=`"$linker_flags`""

# Find the full path of cmake.exe
$cmake_command = Get-Command -CommandType Application cmake
$cmake_path = $cmake_command.Path

Install-Pybind -cmake_path $cmake_path -src_root $ort_src_root -build_config $build_config  -cmake_extra_args $cmake_extra_args

Install-Protobuf -cmake_path $cmake_path -src_root $ort_src_root -build_config $build_config -cmake_extra_args $cmake_extra_args

# TODO: parse it from deps.txt
$protobuf_version="3.18.3"

# ONNX doesn't allow us to specify CMake's path
Install-ONNX -build_config $build_config -src_root $ort_src_root -protobuf_version $protobuf_version