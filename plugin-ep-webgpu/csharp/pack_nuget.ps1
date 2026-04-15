<#
.SYNOPSIS
  Builds the Microsoft.ML.OnnxRuntime.EP.WebGpu NuGet package.

.DESCRIPTION
  Stages native binaries from build artifacts into the runtimes/ layout expected by the
  .csproj and runs 'dotnet pack' to produce the .nupkg and .snupkg files.

  Can be invoked locally or from CI. In CI, pass -ArtifactsDir to point at the
  downloaded pipeline artifacts. Locally, pass individual -BinaryDir_* parameters
  or place binaries manually in the runtimes/ folders.

.EXAMPLE
  # Local: pack win-x64 only from a local build
  .\pack_nuget.ps1 -Version 1.26.0-dev -BinaryDir_WinX64 ..\..\build\webgpu.plugin\Release\Release

.EXAMPLE
  # CI: pack all platforms from downloaded artifacts
  .\pack_nuget.ps1 -Version $(PluginPackageVersion) -ArtifactsDir $(Build.BinariesDirectory)\artifacts -OutputDir $(Build.ArtifactStagingDirectory)\nuget
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Version,

    [string]$OutputDir = (Join-Path $PSScriptRoot "nuget_output"),

    [string]$Configuration = "Release",

    # CI mode: root directory containing per-platform subdirectories (win_x64/bin/, win_arm64/bin/, etc.)
    [string]$ArtifactsDir,

    # Local mode: individual binary directories per platform (takes precedence over ArtifactsDir for that platform)
    [string]$BinaryDir_WinX64,
    [string]$BinaryDir_WinArm64,
    [string]$BinaryDir_LinuxX64,
    [string]$BinaryDir_OsxArm64
)

$ErrorActionPreference = 'Stop'

$projectDir = Join-Path $PSScriptRoot "Microsoft.ML.OnnxRuntime.EP.WebGpu"
$csproj = Join-Path $projectDir "Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj"

if (-not (Test-Path $csproj)) {
    Write-Error "Project file not found: $csproj"
    exit 1
}

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Stage into a temporary directory under the output dir so we don't modify the source tree.
$stagingDir = Join-Path $OutputDir "_staging"
if (Test-Path $stagingDir) {
    Remove-Item -Recurse -Force $stagingDir
}
New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null

# Copy project sources to staging
Write-Host "Staging project files to $stagingDir"
Copy-Item -Path (Join-Path $projectDir "*") -Destination $stagingDir -Recurse -Force
$stagedCsproj = Join-Path $stagingDir "Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj"

# --- Platform definitions ---
$platforms = [ordered]@{
    win_x64 = @{
        rid   = 'win-x64'
        files = @('onnxruntime_providers_webgpu.dll', 'dxil.dll', 'dxcompiler.dll')
        param = $BinaryDir_WinX64
    }
    win_arm64 = @{
        rid   = 'win-arm64'
        files = @('onnxruntime_providers_webgpu.dll', 'dxil.dll', 'dxcompiler.dll')
        param = $BinaryDir_WinArm64
    }
    linux_x64 = @{
        rid   = 'linux-x64'
        files = @('libonnxruntime_providers_webgpu.so')
        param = $BinaryDir_LinuxX64
    }
    macos_arm64 = @{
        rid   = 'osx-arm64'
        files = @('libonnxruntime_providers_webgpu.dylib')
        param = $BinaryDir_OsxArm64
    }
}

# --- Stage binaries ---
$anyStaged = $false

foreach ($entry in $platforms.GetEnumerator()) {
    $name  = $entry.Key
    $info  = $entry.Value
    $rid   = $info.rid

    # Resolve source directory: explicit param > ArtifactsDir > skip
    $sourceDir = $info.param
    if (-not $sourceDir -and $ArtifactsDir) {
        $candidate = Join-Path $ArtifactsDir "$name\bin"
        if (Test-Path $candidate) {
            $sourceDir = $candidate
        }
    }

    if (-not $sourceDir) {
        Write-Host "Skipping $name (no binary directory provided)"
        continue
    }

    if (-not (Test-Path $sourceDir)) {
        Write-Error "Binary directory does not exist: $sourceDir"
        exit 1
    }

    $targetDir = Join-Path $stagingDir "runtimes\$rid\native"
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

    Write-Host "Staging $name -> runtimes/$rid/native/"
    foreach ($file in $info.files) {
        $src = Join-Path $sourceDir $file
        if (-not (Test-Path $src)) {
            Write-Error "Expected binary not found: $src"
            exit 1
        }
        Copy-Item -Path $src -Destination $targetDir -Force
        Write-Host "  $file"
    }
    $anyStaged = $true
}

if (-not $anyStaged) {
    Write-Error "No platform binaries were staged. Provide at least one -BinaryDir_* parameter or -ArtifactsDir."
    exit 1
}

Write-Host ""
Write-Host "Runtimes layout:"
Get-ChildItem -Recurse (Join-Path $stagingDir "runtimes") | ForEach-Object { Write-Host "  $($_.FullName)" }

# --- Pack ---
Write-Host ""
Write-Host "Running dotnet pack (Version=$Version, Configuration=$Configuration)..."

dotnet pack $stagedCsproj --configuration $Configuration "-p:Version=$Version" --output $OutputDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "dotnet pack failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# --- Verify ---
Write-Host ""
$nupkgs = Get-ChildItem "$OutputDir\*.nupkg"
if ($nupkgs.Count -eq 0) {
    Write-Error "No .nupkg files found in $OutputDir"
    exit 1
}
foreach ($pkg in $nupkgs) {
    Write-Host "Produced: $($pkg.Name) ($([math]::Round($pkg.Length / 1MB, 2)) MB)"
}
$snupkgs = Get-ChildItem "$OutputDir\*.snupkg" -ErrorAction SilentlyContinue
foreach ($pkg in $snupkgs) {
    Write-Host "Produced: $($pkg.Name) ($([math]::Round($pkg.Length / 1MB, 2)) MB)"
}

# --- Clean up staging directory ---
if (Test-Path $stagingDir) {
    Remove-Item -Recurse -Force $stagingDir
    Write-Host ""
    Write-Host "Cleaned up staging directory."
}

Write-Host ""
Write-Host "Done. Output: $OutputDir"
