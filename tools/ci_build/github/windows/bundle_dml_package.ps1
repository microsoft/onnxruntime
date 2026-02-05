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

# Debug: List contents of extracted arm64 zip
Write-Host "Contents of $arm64ExtractPath (recursive):"
Get-ChildItem -Path $arm64ExtractPath -Recurse | ForEach-Object { Write-Host "  - $($_.FullName)" }

# 2. Find the target NuGet package.
# It finds all .nupkg files that do not contain "Managed" in their name.
$nupkgFiles = Get-ChildItem -Path . -Filter *.nupkg | Where-Object { ($_.Name -notlike "*Managed*") -and ($_.Name -notlike "*.symbols.nupkg") }

Write-Host "Found $($nupkgFiles.Count) candidate nupkg file(s) for bundling:"
$nupkgFiles | ForEach-Object { Write-Host "  - $($_.FullName)" }

# 3. Select the best package (shortest name prefers Release over Dev, and Main over Symbols)
if ($nupkgFiles.Count -eq 0) {
    Write-Error "Error: No matching NuGet packages found to bundle into."
    exit 1
}
if ($nupkgFiles.Count -gt 1) {
    Write-Warning "Found multiple packages. Selecting the one with the shortest filename as the target for bundling."
}
$nupkg = $nupkgFiles | Sort-Object {$_.Name.Length} | Select-Object -First 1
Write-Host "Selected target package: $($nupkg.Name)"

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

# Debug: Print the .nuspec content
$nuspecFile = Get-ChildItem -Path $tempDir -Filter *.nuspec | Select-Object -First 1
if ($nuspecFile) {
    Write-Host "Found manifest: $($nuspecFile.FullName)"
    Write-Host "--- Manifest Content ---"
    Get-Content $nuspecFile.FullName | ForEach-Object { Write-Host $_ }
    Write-Host "------------------------"
}

# Debug: List contents of extracted target nupkg
Write-Host "Contents of $tempDir (recursive):"
Get-ChildItem -Path $tempDir -Recurse | ForEach-Object { Write-Host "  - $($_.FullName)" }

# Step B: Create the new runtime directory structure.
$newRuntimePath = Join-Path $tempDir "runtimes\win-arm64\native"
Write-Host "Ensuring destination path exists: $newRuntimePath"
New-Item -ItemType Directory -Path $newRuntimePath -Force | Out-Null

# Step C: Copy the ARM64 binaries into the new structure.
$arm64SourcePath = Join-Path . "$arm64ExtractPath\runtimes\win-arm64\native"
if (Test-Path $arm64SourcePath) {
    Write-Host "Copying ARM64 binaries from $arm64SourcePath to $newRuntimePath..."
    $filesToCopy = Get-ChildItem -Path "$arm64SourcePath\*"
    Write-Host "Files found in source: $($filesToCopy.Count)"
    $filesToCopy | ForEach-Object { Write-Host "  -> $($_.Name)" }
    Copy-Item -Path "$arm64SourcePath\*" -Destination $newRuntimePath -Recurse -Force
} else {
    Write-Error "Error: ARM64 source path not found: $arm64SourcePath. Bailing out to avoid creating a broken package."
    exit 1
}

# Step D: Delete the original nupkg file.
Remove-Item -Path $nupkg.FullName -Force

# Step E: Re-compress the modified contents back into a new nupkg file.
Write-Host "Creating new nupkg file at $($nupkg.FullName)..."
Push-Location $tempDir
& $sevenZipPath a -tzip "$($nupkg.FullName)" ".\" -r
Pop-Location

# Debug: Check final nupkg existence
if (Test-Path $nupkg.FullName) {
    Write-Host "Final package created successfully: $($nupkg.FullName)"
    $finalSize = (Get-Item $nupkg.FullName).Length
    Write-Host "Final package size: $finalSize bytes"
}

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
