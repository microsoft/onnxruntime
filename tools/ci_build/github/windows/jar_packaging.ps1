# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Stop the script immediately if any command fails
$ErrorActionPreference = "Stop"

# --- Configuration ---
# Define the target directory.
$TargetDirectory = "$Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64"

# --- Find Version from JAR file ---
Write-Output "Discovering version from JAR files in '$TargetDirectory'..."
$mainJarFile = Get-ChildItem -Path $TargetDirectory -Filter *.jar | Where-Object { 
    $_.Name -notlike '*-sources.jar' -and $_.Name -notlike '*-javadocs.jar' 
} | Select-Object -First 1

if (-not $mainJarFile) {
    throw "Could not find a main JAR file in '$TargetDirectory' to determine the version."
}

# Parse the version from the filename. E.g., "onnxruntime-1.15.0" -> "1.15.0"
$Version = $mainJarFile.BaseName -split '-' | Select-Object -Last 1

if (-not $Version) {
    throw "Failed to parse version from filename '$($mainJarFile.Name)'."
}

Write-Output "Version discovered: $Version"

# Define final archive paths using the discovered version.
$MainArchiveFile = "$TargetDirectory\onnxruntime-$Version.jar"
$TestArchiveFile = "$TargetDirectory\testing.jar"

# --- Reusable Function ---
# This function processes a single platform directory, handling all logic and error checking.
function Process-PlatformArchive {
    param(
        [Parameter(Mandatory=$true)]
        [string]$PlatformPath,

        [string]$CustomLibFile,
        
        [bool]$ArchiveCustomLib = $false
    )

    Write-Output "Processing platform: $PlatformPath..."
    
    # Check if the platform directory exists before trying to enter it.
    if (-not (Test-Path -Path $PlatformPath -PathType Container)) {
        throw "Directory not found: $PlatformPath"
    }

    # Use a try/finally block to guarantee that popd is always called, even if an error occurs.
    try {
        pushd $PlatformPath

        # Handle the custom library file if it exists.
        if ($CustomLibFile -and (Test-Path $CustomLibFile)) {
            
            # Optionally archive the custom library to the test JAR.
            if ($ArchiveCustomLib) {
                Write-Output "  -> Archiving '$CustomLibFile' to test JAR..."
                7z a $TestArchiveFile $CustomLibFile
                if ($LASTEXITCODE -ne 0) { throw "7z failed to archive '$CustomLibFile' with exit code $LASTEXITCODE." }
            }
            
            # Always remove the custom library file after processing.
            Write-Output "  -> Removing '$CustomLibFile'..."
            Remove-Item -Path $CustomLibFile
        }

        # Archive the remaining contents to the main JAR.
        Write-Output "  -> Archiving contents to main JAR..."
        7z a $MainArchiveFile .
        if ($LASTEXITCODE -ne 0) { throw "7z failed to archive contents of '$PlatformPath' with exit code $LASTEXITCODE." }

    }
    finally {
        popd
        Write-Output "Finished platform: $PlatformPath"
        Write-Output "--------------------------------"
    }
}

# --- Main Execution ---
# Define all platforms and their specific properties in an array.
$platforms = @(
    [pscustomobject]@{ Path = 'onnxruntime-java-linux-x64';      Lib = 'libcustom_op_library.so';   ArchiveLib = $true  },
    [pscustomobject]@{ Path = 'onnxruntime-java-osx-x86_64';     Lib = 'libcustom_op_library.dylib';ArchiveLib = $true  },
    [pscustomobject]@{ Path = 'onnxruntime-java-linux-aarch64';  Lib = 'libcustom_op_library.so';   ArchiveLib = $false },
    [pscustomobject]@{ Path = 'onnxruntime-java-osx-arm64';      Lib = 'libcustom_op_library.dylib';ArchiveLib = $false }
)

# Loop through the defined platforms and process each one using the function.
foreach ($platform in $platforms) {
    Process-PlatformArchive -PlatformPath $platform.Path -CustomLibFile $platform.Lib -ArchiveCustomLib $platform.ArchiveLib
}

Write-Output "All archives updated successfully."