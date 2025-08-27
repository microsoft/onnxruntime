# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget-Java Packaging Pipeline

# Define the directory for NuGet artifacts.
$nuget_artifacts_dir = "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
# Create the directory if it doesn't exist.
New-Item -Path $nuget_artifacts_dir -ItemType directory -ErrorAction SilentlyContinue

## .zip files
# Unzip files directly, excluding the iOS xcframework to preserve its symlinks.
Get-ChildItem -Path "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact\*" -Include *.zip -Exclude onnxruntime_ios_xcframework.*.zip |
Foreach-Object {
    # The -snld20 flag is used to bypass security checks for creating symbolic links (added in 7-Zip 25.01).
    $arguments = "x", "$($_.FullName)", "-y", "-o$nuget_artifacts_dir", "-snld20"
    Write-Output "Executing: 7z.exe $arguments"
    # Directly call 7z.exe using the call operator '&'
    & 7z.exe $arguments
    # Check the exit code of the last command. A non-zero code indicates an error.
    if ($LASTEXITCODE -ne 0) {
        throw "Error extracting '$($_.FullName)'. Exit code: $LASTEXITCODE"
    }
}

## .tgz files
# First, extract the .tar file from the .tgz archive.
Get-ChildItem "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" -Filter *.tgz |
Foreach-Object {
    # The -snld20 flag is used to bypass security checks for creating symbolic links (added in 7-Zip 25.01).
    $arguments = "x", "$($_.FullName)", "-y", "-o$Env:BUILD_BINARIESDIRECTORY\nuget-artifact", "-snld20"
    Write-Output "Executing: 7z.exe $arguments"
    & 7z.exe $arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Error extracting '$($_.FullName)'. Exit code: $LASTEXITCODE"
    }
}

# Now, extract the contents from the .tar file into the final directory.
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

# Process iOS xcframework
$xcframeworks = Get-ChildItem "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" -Filter onnxruntime_ios_xcframework.*.zip
if ($xcframeworks.Count -eq 1) {
    $xcframework = $xcframeworks[0]
    $target_dir = "$nuget_artifacts_dir\onnxruntime-ios-xcframework"
    # Use the required filename format, removing version info.
    $target_file = "$target_dir\onnxruntime.xcframework.zip"
    New-Item -Path $target_dir -ItemType directory -ErrorAction SilentlyContinue

    Write-Output "Copying $($xcframework.FullName) to $target_file"
    Copy-Item $xcframework.FullName $target_file
}
elseif ($xcframeworks.Count -gt 1) {
    Write-Error "Expected at most one onnxruntime_ios_xcframework*.zip file but got: [$xcframeworks]"
}

# Copy Android AAR file.
# There should only be one .aar file for a full build.
$aars = Get-ChildItem "$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" -Filter *.aar
if ($aars.Count -eq 1) {
    $aar = $aars[0]
    $aar_prefix = "onnxruntime"
    if ($aar.Name -like "onnxruntime-training*") {
        $aar_prefix = "onnxruntime-training"
    }
    $target_dir = "$nuget_artifacts_dir\$aar_prefix-android-aar"
    # Remove version info from the filename for consistency.
    $target_file = "$target_dir\onnxruntime.aar"
    New-Item -Path $target_dir -ItemType directory -ErrorAction SilentlyContinue

    Write-Output "Copying $($aar.FullName) to $target_file"
    Copy-Item $aar.FullName $target_file
}
elseif ($aars.Count -gt 1) {
    Write-Error "Expected at most one Android .aar file but got: [$aars]"
}

# Check if this is a training pipeline by looking for a specific directory.
$is_training_pipeline = Test-Path -Path "$nuget_artifacts_dir\onnxruntime-training-win-x64-*"
if ($is_training_pipeline) {
    Write-Output "onnxruntime-training-win-x64-* dir exists. This is a training pipeline."
}

# Copy onnxruntime and protoc binaries required by tests.
$destinationDir = "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo"
if ($is_training_pipeline) {
    Copy-Item -Path "$nuget_artifacts_dir\onnxruntime-training-win-x64-*\lib\*" -Destination $destinationDir -Recurse
}
else {
    Copy-Item -Path "$nuget_artifacts_dir\onnxruntime-win-x64-*\lib\*" -Destination $destinationDir -Recurse
}

# Rename directories to remove the architecture-specific suffix.
Write-Output "Renaming onnxruntime directories..."
Get-ChildItem -Directory -Path "$nuget_artifacts_dir\onnxruntime-*" | ForEach-Object {
    $dirname = $_.Name
    # Find the last hyphen and remove the suffix.
    $lastHyphenIndex = $dirname.LastIndexOf('-')
    if ($lastHyphenIndex -gt -1) {
        $newName = $dirname.Substring(0, $lastHyphenIndex)
        $newPath = Join-Path -Path $_.Parent.FullName -ChildPath $newName
        Write-Output "Renaming '$($_.FullName)' to '$newPath'"
        Rename-Item -Path $_.FullName -NewName $newName
    }
}

# List the final artifacts.
Write-Output "Post-copy artifacts:"
Get-ChildItem -Recurse $nuget_artifacts_dir

# Check if all .so files are symlinks, which is required for the package structure.
Write-Output "Checking for .so file symlinks..."
$so_files = Get-ChildItem -Recurse -Path $nuget_artifacts_dir -Filter *.so
# Find any .so files that are NOT symlinks.
$non_symlink_so_files = $so_files | Where-Object { -not ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) }

if ($non_symlink_so_files) {
    Write-Error "Found .so files that are not symlinks:"
    foreach ($file in $non_symlink_so_files) {
        Write-Error "- $($file.FullName)"
    }
    throw "One or more .so files are not symlinks. This can cause issues in the NuGet package."
}
else {
    Write-Output "All found .so files are correctly configured as symlinks."
}
