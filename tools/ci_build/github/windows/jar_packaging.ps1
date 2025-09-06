# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('cpu', 'gpu')]
    [string]$PackageType = 'cpu'
)

# Stop the script immediately if any command fails
$ErrorActionPreference = "Stop"

# --- Configuration ---
# Base directory where all the unzipped platform-specific artifacts are.
$ArtifactsBaseDirectory = "$Env:BUILD_BINARIESDIRECTORY\java-artifact"
# This is the directory of the main Windows CPU package, which we will use as the
# destination for all updated archives and for discovering the version.
$PrimaryPackageDirectory = Join-Path $ArtifactsBaseDirectory "onnxruntime-java-win-x64"

# --- Version Discovery ---
# We discover the version from the primary CPU package. All other packages (like GPU)
# will be updated to match this version.
Write-Output "Discovering version from JAR files in '$PrimaryPackageDirectory'..."
$mainCpuJarFile = Get-ChildItem -Path $PrimaryPackageDirectory -Filter "onnxruntime-*.jar" | Where-Object {
    $_.Name -notlike '*-sources.jar' -and $_.Name -notlike '*-javadocs.jar'
} | Select-Object -First 1

if (-not $mainCpuJarFile) {
    throw "Could not find a main JAR file in '$PrimaryPackageDirectory' to determine the version."
}

# Parse the version from the filename. E.g., "onnxruntime-1.15.0" -> "1.15.0"
$Version = ($mainCpuJarFile.BaseName -split '-')[-1]

if (-not $Version) {
    throw "Failed to parse version from filename '$($mainCpuJarFile.Name)'."
}

Write-Output "Version discovered: $Version"

# --- Reusable Function ---
# This function processes a single platform directory, handling all archiving logic.
function Process-PlatformArchive {
    param(
        [Parameter(Mandatory=$true)]
        [string]$PlatformPath,
        [Parameter(Mandatory=$true)]
        [string]$MainArchiveFile,
        [Parameter(Mandatory=$true)]
        [string]$TestArchiveFile,
        [string]$CustomLibFile,
        [bool]$ArchiveCustomLib = $false
    )

    Write-Output "Processing platform: $PlatformPath..."

    if (-not (Test-Path -Path $PlatformPath -PathType Container)) {
        throw "Directory not found: $PlatformPath"
    }

    # Use a try/finally block to guarantee that popd is always called.
    try {
        pushd $PlatformPath

        # 1. Handle the custom library file if it exists.
        if ($CustomLibFile -and (Test-Path $CustomLibFile)) {
            if ($ArchiveCustomLib) {
                Write-Output "  -> Archiving '$CustomLibFile' to test JAR..."
                7z a $TestArchiveFile $CustomLibFile
                if ($LASTEXITCODE -ne 0) { throw "7z failed to archive '$CustomLibFile' with exit code $LASTEXITCODE." }
            }
            Write-Output "  -> Removing '$CustomLibFile'..."
            Remove-Item -Path $CustomLibFile
        }

        # 2. Archive the remaining contents to the main JAR.
        Write-Output "  -> Archiving all contents to main JAR '$((Split-Path $MainArchiveFile -Leaf))'..."
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
# Define the package to be processed based on the script argument.
$packageDefinition = $null

if ($PackageType -eq 'cpu') {
    Write-Output "## Configuring for CPU package build..."
    $packageDefinition = [pscustomobject]@{
        PackageName = 'onnxruntime'
        Platforms = @(
            [pscustomobject]@{ Path = 'onnxruntime-java-linux-x64';      Lib = 'libcustom_op_library.so';    ArchiveLib = $true  },
            [pscustomobject]@{ Path = 'onnxruntime-java-osx-x86_64';     Lib = 'libcustom_op_library.dylib'; ArchiveLib = $true  },
            [pscustomobject]@{ Path = 'onnxruntime-java-linux-aarch64';  Lib = 'libcustom_op_library.so';    ArchiveLib = $false },
            [pscustomobject]@{ Path = 'onnxruntime-java-osx-arm64';      Lib = 'libcustom_op_library.dylib'; ArchiveLib = $false }
        )
    }
}
elseif ($PackageType -eq 'gpu') {
    Write-Output "## Configuring for GPU package build..."
    $packageDefinition = [pscustomobject]@{
        PackageName = 'onnxruntime_gpu'
        Platforms = @(
            [pscustomobject]@{
                Path = 'onnxruntime-java-linux-x64-tensorrt'
                Lib = 'libcustom_op_library.so'
                ArchiveLib = $false # Assuming custom ops aren't needed for the GPU test jar
                GpuLibs = @( # List of GPU-specific libraries to copy
                    [pscustomobject]@{
                        Source = (Join-Path $ArtifactsBaseDirectory 'onnxruntime-java-linux-x64\ai\onnxruntime\native\linux-x64\libonnxruntime_providers_cuda.so')
                        Destination = 'ai\onnxruntime\native\linux-x64'
                    },
                    [pscustomobject]@{
                        Source = (Join-Path $ArtifactsBaseDirectory 'onnxruntime-java-linux-x64\ai\onnxruntime\native\linux-x64\libonnxruntime_providers_tensorrt.so')
                        Destination = 'ai\onnxruntime\native\linux-x64'
                    }
                )
            },
            [pscustomobject]@{
                Path = 'onnxruntime-java-win-x64-gpu' # Example path for Windows GPU
                Lib = 'custom_op_library.dll' # Example custom lib name
                ArchiveLib = $false
                GpuLibs = @(
                    [pscustomobject]@{
                        Source = (Join-Path $ArtifactsBaseDirectory 'onnxruntime-java-win-x64\ai\onnxruntime\native\win-x64\onnxruntime_providers_cuda.dll')
                        Destination = 'ai\onnxruntime\native\win-x64'
                    },
                     [pscustomobject]@{
                        Source = (Join-Path $ArtifactsBaseDirectory 'onnxruntime-java-win-x64\ai\onnxruntime\native\win-x64\onnxruntime_providers_tensorrt.dll')
                        Destination = 'ai\onnxruntime\native\win-x64'
                    }
                )
            }
        )
    }
} else {
    throw "Invalid PackageType specified: $PackageType"
}

# --- Processing Loop ---
# The logic now processes the single package definition determined by the script argument.
$package = $packageDefinition
Write-Output "## Processing Package: $($package.PackageName)"

# Define the final archive paths for this specific package. All archives are placed
# in the primary windows directory for easy collection.
$mainArchiveFile = Join-Path $PrimaryPackageDirectory "$($package.PackageName)-$Version.jar"
$testArchiveFile = Join-Path $PrimaryPackageDirectory "testing-$($package.PackageName).jar"

# Loop through each platform within the package
foreach ($platform in $package.Platforms) {
    $platformFullPath = Join-Path $ArtifactsBaseDirectory $platform.Path

    # --- GPU Pre-processing Step ---
    # If the platform has GPU libraries defined, copy them over first.
    if ($platform.PSObject.Properties['GpuLibs']) {
        foreach ($gpuLib in $platform.GpuLibs) {
            $destDir = Join-Path $platformFullPath $gpuLib.Destination
            Write-Output "  -> Copying GPU library '$((Split-Path $gpuLib.Source -Leaf))' to '$destDir'"
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
            Copy-Item -Path $gpuLib.Source -Destination $destDir
        }
    }

    # Call the main processing function
    Process-PlatformArchive -PlatformPath $platformFullPath `
                            -MainArchiveFile $mainArchiveFile `
                            -TestArchiveFile $testArchiveFile `
                            -CustomLibFile $platform.Lib `
                            -ArchiveCustomLib $platform.ArchiveLib
}

Write-Output "All archives updated successfully."