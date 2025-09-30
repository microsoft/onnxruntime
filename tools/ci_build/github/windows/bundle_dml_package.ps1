# Description:
# This script modifies a specific .nupkg file by injecting ARM64 binaries.
# It's designed to be called from an Azure DevOps pipeline where the
# working directory is set to the location of the NuGet packages.
#
# Arguments:
#   -ArtifactStagingDirectory: The path to the build's artifact staging directory,
#     passed from the Azure DevOps pipeline.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactStagingDirectory
)

# --- Configuration ---
# Define the path to the 7-Zip executable.
$sevenZipPath = "C:\Program Files\7-Zip\7z.exe"

# --- Script Body ---
Write-Host "Script started. Current working directory: $(Get-Location)"
Write-Host "Artifact Staging Directory: $ArtifactStagingDirectory"

# 1. Unzip the supplemental ARM64 binaries.
$arm64ZipFile = "win-dml-arm64.zip"
$arm64ExtractPath = "win-dml-arm64-unzipped"
Write-Host "Extracting $arm64ZipFile to $arm64ExtractPath..."
& $sevenZipPath x $arm64ZipFile -o"$arm64ExtractPath" -y

# 2. Find the target NuGet package.
# It finds all .nupkg files that do not contain "Managed" in their name.
$nupkgFiles = Get-ChildItem -Path . -Recurse -Filter *.nupkg | Where-Object { $_.Name -notlike "*Managed*" }

# 3. Validate that exactly one package was found.
if ($nupkgFiles.Count -ne 1) {
    Write-Error "Error: Expected to find exactly one non-managed NuGet package, but found $($nupkgFiles.Count)."
    exit 1
}
$nupkg = $nupkgFiles[0]
Write-Host "Found package to process: $($nupkg.Name)"

# 4. Validate the package name matches the expected format.
if ($nupkg.Name -notlike "Microsoft.ML.OnnxRuntime.DirectML*.nupkg") {
    Write-Error "Error: Package name '$($nupkg.Name)' does not match the expected pattern 'Microsoft.ML.OnnxRuntime.DirectML*.nupkg'."
    exit 1
}
Write-Host "Package name validation passed."

# --- Package Modification ---
Write-Host "---"
Write-Host "Processing package: $($nupkg.FullName)"

# Create a temporary directory for modification.
$tempDir = Join-Path $nupkg.DirectoryName ($nupkg.BaseName + "_temp")
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Step A: Extract the original nupkg to the temporary directory.
Write-Host "Extracting $($nupkg.Name) to $tempDir..."
& $sevenZipPath x $nupkg.FullName -o"$tempDir" -y

# Step B: Create the new runtime directory structure.
$newRuntimePath = Join-Path $tempDir "runtimes\win-arm64\native"
New-Item -ItemType Directory -Path $newRuntimePath -Force | Out-Null

# Step C: Copy the ARM64 binaries into the new structure.
$arm64SourcePath = Join-Path . "$arm64ExtractPath\runtimes\win-arm64\native"
Write-Host "Copying ARM64 binaries from $arm64SourcePath to $newRuntimePath..."
Copy-Item -Path "$arm64SourcePath\*" -Destination $newRuntimePath -Recurse -Force

# Step D: Delete the original nupkg file.
Remove-Item -Path $nupkg.FullName -Force

# Step E: Re-compress the modified contents back into a new nupkg file.
Write-Host "Creating new nupkg file at $($nupkg.FullName)..."
Push-Location $tempDir
& $sevenZipPath a -tzip "$($nupkg.FullName)" ".\" -r
Pop-Location

# --- Cleanup and Final Steps ---
Write-Host "Cleaning up temporary directory $tempDir..."
Remove-Item -Recurse -Force $tempDir

Write-Host "Cleaning up temporary ARM64 directory $arm64ExtractPath..."
Remove-Item -Recurse -Force $arm64ExtractPath

# Copy the final package to the artifact staging directory.
Write-Host "Copying final artifact to $ArtifactStagingDirectory..."
Copy-Item -Path ".\Microsoft.ML.OnnxRuntime.DirectML*.nupkg" -Destination $ArtifactStagingDirectory -Force

Write-Host "---"
Write-Host "Script completed successfully."