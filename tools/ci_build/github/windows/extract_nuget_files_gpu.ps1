# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget-Java Packaging Pipeline
# Define the directory for NuGet artifacts.
$nuget_artifacts_dir = "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
# Create the directory if it doesn't exist.
New-Item -Path $nuget_artifacts_dir -ItemType directory -ErrorAction SilentlyContinue

## .zip files
Get-ChildItem "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" -Filter *.zip |
Foreach-Object {
    # The -snld20 flag is used to bypass security checks for creating symbolic links (added in 7-Zip 25.01).
    $arguments = "x", "$($_.FullName)", "-y", "-o$nuget_artifacts_dir", "-snld20"
    Write-Output "Executing: 7z.exe $arguments"
    & 7z.exe $arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Error extracting '$($_.FullName)'. Exit code: $LASTEXITCODE"
    }
}

## .tgz files
Get-ChildItem "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" -Filter *.tgz |
Foreach-Object {
    # The -snld20 flag is used to bypass security checks for creating symbolic links (added in 7-Zip 25.01).
    # *.tar will be created after *.tgz is extracted
    $arguments = "x", "$($_.FullName)", "-y", "-o$Env:BUILD_BINARIESDIRECTORY\nuget-artifact", "-snld20"
    Write-Output "Executing: 7z.exe $arguments"
    & 7z.exe $arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Error extracting '$($_.FullName)'. Exit code: $LASTEXITCODE"
    }
}

## .tar files
Get-ChildItem "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" -Filter *.tar |
Foreach-Object {
    # The -snld20 flag is used to bypass security checks for creating symbolic links (added in 7-Zip 25.01).
    $arguments = "x", "$($_.FullName)", "-y", "-o$nuget_artifacts_dir", "-snld20"
    Write-Output "Executing: 7z.exe $arguments"
    & 7z.exe $arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Error extracting '$($_.FullName)'. Exit code: $LASTEXITCODE"
    }
}

# Create directory for protobuf build dependencies.
New-Item -Path "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build\RelWithDebInfo" -ItemType directory -ErrorAction SilentlyContinue

# Copy CUDA libraries.
Copy-Item -Path "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-x64-cuda-*\lib\*" -Destination "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo"

# Install protoc via dotnet.
$protocInstallDir = "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build"
dotnet new console
dotnet add package Google.Protobuf.Tools --version 3.21.12 --package-directory $protocInstallDir
if ($LASTEXITCODE -ne 0) {
    throw "Error adding Google.Protobuf.Tools package. Exit code: $LASTEXITCODE"
}

# Find and copy the protoc executable.
$protocDir = Get-ChildItem -Path $protocInstallDir -Recurse -Filter "protoc.exe" | Select-Object -ExpandProperty DirectoryName -First 1
if ($protocDir) {
    Write-Output "Found protoc directory: $protocDir"
    Copy-Item -Path $protocDir -Destination "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build\RelWithDebInfo"
}
else {
    Write-Error "Could not find protoc.exe in $protocInstallDir"
}

# Rename onnxruntime directories to a generic format.
$ort_dirs = Get-ChildItem -Path "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-*" -Directory
foreach ($ort_dir in $ort_dirs) {
    $dirname = Split-Path -Path $ort_dir -Leaf
    $lastHyphenIndex = $dirname.LastIndexOf('-')
    if ($lastHyphenIndex -gt -1) {
        $newName = $dirname.Substring(0, $lastHyphenIndex)
        $newPath = Join-Path -Path $ort_dir.Parent.FullName -ChildPath $newName
        Write-Output "Renaming '$($ort_dir.FullName)' to '$newPath'"
        Rename-Item -Path $ort_dir.FullName -NewName $newName
    }
}
