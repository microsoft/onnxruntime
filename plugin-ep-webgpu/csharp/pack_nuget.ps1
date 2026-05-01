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
    [string]$BinaryDir_OsxArm64,

    # Optional NuGet.config to pass to `dotnet pack` via --configfile.
    [string]$NuGetConfig,

    # Optional explicit staging directory. If unset, a directory under $OutputDir is used.
    # Useful in CI when the caller wants to point ESRP signing at a known location.
    [string]$StagingDir,

    # Split-phase switches for CI signing: build the managed DLL in one task
    # (so it can be ESRP-signed), then pack with --no-build in a later task.
    # When neither is set, the script does the full build+pack end-to-end.
    [switch]$BuildOnly,
    [switch]$PackOnly,

    # Optional list of platforms that MUST be staged successfully. CI passes the set
    # of enabled platforms (derived from the pipeline parameters) so that a renamed
    # or missing upstream artifact fails fast instead of silently producing a partial
    # multi-RID package. When omitted (typical for local dev) the script just requires
    # at least one platform to be staged.
    # Accepted as a comma-separated string for compatibility with how Azure Pipelines
    # expands variables into PowerShell arguments.
    [string]$RequiredPlatforms
)

if ($BuildOnly -and $PackOnly) {
    Write-Error "-BuildOnly and -PackOnly are mutually exclusive."
    exit 1
}

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

$requiredPlatformList = @()
if ($RequiredPlatforms) {
    $requiredPlatformList = $RequiredPlatforms.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    $invalid = @($requiredPlatformList | Where-Object { $_ -notin $platforms.Keys })
    if ($invalid.Count -gt 0) {
        Write-Error "Unknown platform(s) in -RequiredPlatforms: $($invalid -join ', '). Valid: $($platforms.Keys -join ', ')."
        exit 1
    }
}

if ($NuGetConfig -and -not (Test-Path $NuGetConfig)) {
    Write-Error "NuGet.config not found: $NuGetConfig"
    exit 1
}

$ErrorActionPreference = 'Stop'

$projectDir = Join-Path $PSScriptRoot "Microsoft.ML.OnnxRuntime.EP.WebGpu"
$csproj = Join-Path $projectDir "Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj"

if (-not (Test-Path $csproj)) {
    Write-Error "Project file not found: $csproj"
    exit 1
}

# Resolve the minimum-required ORT version file to an absolute path. Passed to MSBuild
# via -p:OnnxRuntimeMinVersionFile so the staged project (which builds out of a copy
# under $StagingDir) can still find the file in the original source tree.
$ortMinVersionFile = Join-Path $PSScriptRoot '..\MIN_ONNXRUNTIME_VERSION'
if (-not (Test-Path $ortMinVersionFile)) {
    Write-Error "MIN_ONNXRUNTIME_VERSION file not found: $ortMinVersionFile"
    exit 1
}
$ortMinVersionFile = (Resolve-Path $ortMinVersionFile).Path

$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Stage into a temporary directory so we don't modify the source tree.
# When the caller provides $StagingDir explicitly, they own its lifecycle (no auto-cleanup).
$ownsStaging = -not $StagingDir
if ($StagingDir) {
    $StagingDir = [System.IO.Path]::GetFullPath($StagingDir)
}
else {
    $StagingDir = Join-Path $OutputDir "_staging"
}
$stagedCsproj = Join-Path $StagingDir "Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj"

if ($PackOnly) {
    if (-not (Test-Path $stagedCsproj)) {
        Write-Error "Staged project not found at $stagedCsproj. Run with -BuildOnly first."
        exit 1
    }
    Write-Host "Reusing existing staging directory: $StagingDir"
}
else {
    if (Test-Path $StagingDir) {
        Remove-Item -Recurse -Force $StagingDir
    }
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

    # Copy project sources to staging. Exclude bin/obj so a stale local build of the
    # in-tree project doesn't get dragged into the staged copy (CI is unaffected since
    # the workspace is clean per run).
    Write-Host "Staging project files to $StagingDir"
    Copy-Item -Path (Join-Path $projectDir "*") -Destination $StagingDir -Recurse -Force -Exclude bin,obj
}

# --- Stage binaries ---
if (-not $PackOnly) {
    $stagedPlatforms = [System.Collections.Generic.HashSet[string]]::new()

    foreach ($entry in $platforms.GetEnumerator()) {
        $name  = $entry.Key
        $info  = $entry.Value
        $rid   = $info.rid
        $isRequired = $requiredPlatformList -and ($requiredPlatformList -contains $name)

        # Resolve source directory: explicit param > ArtifactsDir > skip
        $sourceDir = $info.param
        if (-not $sourceDir -and $ArtifactsDir) {
            $candidate = Join-Path $ArtifactsDir "$name\bin"
            if (Test-Path $candidate) {
                $sourceDir = $candidate
            }
            elseif ($isRequired) {
                Write-Error "Required platform '$name' artifact directory not found: $candidate"
                exit 1
            }
        }

        if (-not $sourceDir) {
            if ($isRequired) {
                Write-Error "Required platform '$name' has no binary directory (pass -BinaryDir_$($name -replace '_','') or -ArtifactsDir)."
                exit 1
            }
            Write-Host "Skipping $name (no binary directory provided)"
            continue
        }

        if (-not (Test-Path $sourceDir)) {
            Write-Error "Binary directory does not exist: $sourceDir"
            exit 1
        }

        $targetDir = Join-Path $StagingDir "runtimes\$rid\native"
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
        [void]$stagedPlatforms.Add($name)
    }

    if ($requiredPlatformList) {
        $missing = @($requiredPlatformList | Where-Object { -not $stagedPlatforms.Contains($_) })
        if ($missing.Count -gt 0) {
            Write-Error "Required platforms not staged: $($missing -join ', ')"
            exit 1
        }
    }
    elseif ($stagedPlatforms.Count -eq 0) {
        Write-Error "No platform binaries were staged. Provide at least one -BinaryDir_* parameter or -ArtifactsDir."
        exit 1
    }

    Write-Host ""
    Write-Host "Runtimes layout:"
    Get-ChildItem -Recurse (Join-Path $StagingDir "runtimes") | ForEach-Object { Write-Host "  $($_.FullName)" }
}

# --- Build / Pack ---
if ($BuildOnly) {
    Write-Host ""
    Write-Host "Running dotnet build (Version=$Version, Configuration=$Configuration)..."

    $buildArgs = @(
        $stagedCsproj,
        '--configuration', $Configuration,
        "-p:Version=$Version",
        "-p:OnnxRuntimeMinVersionFile=$ortMinVersionFile"
    )
    if ($NuGetConfig) {
        $buildArgs += @('--configfile', $NuGetConfig)
        Write-Host "Using NuGet.config: $NuGetConfig"
    }

    dotnet build @buildArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "dotnet build failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    $managedDll = Join-Path $StagingDir "bin\$Configuration\netstandard2.0\Microsoft.ML.OnnxRuntime.EP.WebGpu.dll"
    if (-not (Test-Path $managedDll)) {
        Write-Error "Managed DLL not found after build: $managedDll"
        exit 1
    }
    Write-Host ""
    Write-Host "Built managed DLL: $managedDll"
    Write-Host "Staging directory preserved for subsequent -PackOnly invocation."
    exit 0
}

Write-Host ""
Write-Host "Running dotnet pack (Version=$Version, Configuration=$Configuration)..."

$packArgs = @(
    $stagedCsproj,
    '--configuration', $Configuration,
    "-p:Version=$Version",
    "-p:OnnxRuntimeMinVersionFile=$ortMinVersionFile",
    '--output', $OutputDir
)
if ($PackOnly) {
    $packArgs += '--no-build'
}
if ($NuGetConfig) {
    $packArgs += @('--configfile', $NuGetConfig)
    Write-Host "Using NuGet.config: $NuGetConfig"
}

dotnet pack @packArgs
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
if ($ownsStaging -and (Test-Path $StagingDir)) {
    Remove-Item -Recurse -Force $StagingDir
    Write-Host ""
    Write-Host "Cleaned up staging directory."
}

Write-Host ""
Write-Host "Done. Output: $OutputDir"
