# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script depends on python.exe, cmake.exe and Visual C++ spectre-mitigated libs.
# Please setup AGENT_TEMPDIRECTORY env variable before running this script
# Your PATH must contain a dir that contains python.exe. And cpu arch of the python.exe
# must match the $cpu_arch passed in to this script.

 param (
    [string]$cpu_arch = "x64",
    [string]$build_config = "RelWithDebInfo",
    [string]$install_prefix = ".",
    [switch]$use_cache
 )

. "$PSScriptRoot\helpers.ps1"

$ort_src_root = (Get-Item $PSScriptRoot).parent.parent.parent.parent.FullName

Write-Host "ONNX Runtime src root: $ort_src_root"

$ErrorActionPreference = "Stop"

$Env:Path = "$install_prefix\bin;" + $env:Path
$Env:MSBUILDDISABLENODEREUSE=1

New-Item -Path "$install_prefix" -ItemType Directory -Force

# Setup compile flags
$compile_flags = @('/MP', '/guard:cf', '/DWIN32', '/D_WINDOWS', '/DWINVER=0x0A00', '/D_WIN32_WINNT=0x0A00', '/DNTDDI_VERSION=0x0A000000', '/W3')
$linker_flags=@('/guard:cf')

if ($use_cache) {
  $debug_info_format = "/Z7"
}
else {
  $debug_info_format = "/Zi"
}

if($build_config -eq 'Release'){
  $compile_flags += "/O2", "/Ob2", "/DNDEBUG", "/Gw", "/GL"
} elseif($build_config -eq 'RelWithDebInfo'){
  $compile_flags += "$debug_info_format", "/O2", "/Ob1", "/DNDEBUG", "/Gw", "/GL"
} elseif($build_config -eq 'Debug'){
  $compile_flags += "$debug_info_format", "/Ob0", "/Od", "/RTC1"
} elseif($build_config -eq 'MinSizeRel'){
  $compile_flags += "/O1", "/Ob1", "/DNDEBUG", "/Gw", "/GL"
}

# cmake args that applies to every 3rd-party library
[string[]]$cmake_extra_args="`"-DCMAKE_C_FLAGS=$compile_flags`"", "--compile-no-warning-as-error", "--fresh", "-Wno-dev"


if($cpu_arch -eq 'x86'){
  $cmake_extra_args +=  "-A", "Win32", "-T", "host=x64"
  $compile_flags += '/Qspectre'
  $linker_flags += '/machine:x86'
} elseif($cpu_arch -eq 'x64') {
  $linker_flags += '/machine:x64'
  $compile_flags += '/Qspectre'
} elseif($cpu_arch -eq 'arm') {
  $linker_flags += '/machine:ARM'
} elseif($cpu_arch -eq 'arm64') {
  $linker_flags += '/machine:ARM64'
} elseif($cpu_arch -eq 'arm64ec') {
  $linker_flags += '/machine:ARM64EC'
} else {
  throw "$cpu_arch is not supported"
}

Write-Host $compile_flags

$cmake_extra_args += "-DCMAKE_CXX_STANDARD=17", "`"-DCMAKE_CXX_FLAGS=$compile_flags /EHsc`""

if ($use_cache) {
  if ($build_config -eq 'RelWithDebInfo') {
    $cmake_extra_args += "-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=`"/MD /Z7 /O2 /Ob1 /DNDEBUG`""
  }
  elseif ($build_config -eq 'Debug') {
    $cmake_extra_args += "-DCMAKE_CXX_FLAGS_DEBUG=`"/MDd /Z7 /Ob0 /Od /RTC1`""
  }
}

$cmake_extra_args += "-DCMAKE_EXE_LINKER_FLAGS=`"$linker_flags`""

# Find the full path of cmake.exe
$cmake_command = (Get-Command -CommandType Application cmake)[0]
$cmake_path = $cmake_command.Path
$vshwere_path =  Join-Path -Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if(-not (Test-Path $vshwere_path -PathType Leaf)){
  $vshwere_path =  Join-Path -Path ${env:ProgramFiles} "Microsoft Visual Studio\Installer\vswhere.exe"
}

$msbuild_path = &$vshwere_path -latest -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe | select-object -first 1

Install-Pybind -cmake_path $cmake_path -src_root $ort_src_root -build_config $build_config  -cmake_extra_args $cmake_extra_args -msbuild_path $msbuild_path

Install-Abseil -cmake_path $cmake_path -src_root $ort_src_root -build_config $build_config -cmake_extra_args $cmake_extra_args -msbuild_path $msbuild_path

Install-Protobuf -cmake_path $cmake_path -src_root $ort_src_root -build_config $build_config -cmake_extra_args $cmake_extra_args -msbuild_path $msbuild_path

$protobuf_version="4.21.12"

# ONNX doesn't allow us to specify CMake's path
Install-ONNX -build_config $build_config -src_root $ort_src_root -protobuf_version $protobuf_version
