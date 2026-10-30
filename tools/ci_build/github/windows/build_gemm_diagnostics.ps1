<#
.SYNOPSIS
Builds the exact-base GEMM diagnostic CUDA plugin without running tests or installing it.
.EXAMPLE
.\tools\ci_build\github\windows\build_gemm_diagnostics.ps1 `
    -Python C:\venv\Scripts\python.exe `
    -CMake "C:\Program Files\Microsoft Visual Studio\18\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" `
    -CudaHome C:\CUDA\v13.4 -CudnnHome C:\cuDNN\v9.27 -BuildDirectory C:\b\ort-diag
.NOTES
Use a short checkout and build path. Supply an existing isolated Python environment.
Dependencies use the CI internal asset cache and repository-pinned source archives;
this script does not install Python packages or toolchains.
This is a local toolchain variant: BinSkim/Spectre flags are omitted and cudafe
diagnostic 2803 is suppressed for CUDA 13.4. No QUICK_BUILD reduction is enabled.
Resume with -Mode Build. Do not reconfigure a populated FetchContent tree:
some dependency patch steps are not idempotent. Use a new directory instead.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$CMake,
    [Parameter(Mandatory = $true)][string]$CudaHome,
    [Parameter(Mandatory = $true)][string]$CudnnHome,
    [Parameter(Mandatory = $true)][string]$BuildDirectory,
    [ValidateSet('ARM64', 'x64')][string]$Architecture = 'ARM64',
    [ValidateSet('All', 'Configure', 'Build', 'Plan')][string]$Mode = 'All',
    [string]$Generator = 'Visual Studio 18 2026',
    [ValidateRange(1, 2)][int]$Jobs = 2
)

$ErrorActionPreference = 'Stop'
$diagnosticSource = '64c1a75e1715442956ca40d3c11106a29d0b910e'
$providerBase = '123752d361d2da8ff592b347772f31362e0966bd'
$scriptPath = 'tools/ci_build/github/windows/build_gemm_diagnostics.ps1'
$repo = & git -C $PSScriptRoot rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate the repository.' }
$source = & git -C $repo rev-parse HEAD
$parent = & git -C $repo rev-parse "$diagnosticSource^"
if ($LASTEXITCODE -ne 0 -or $parent -ne $providerBase) {
    throw 'The diagnostic commit must have the exact requested provider parent.'
}
& git -C $repo merge-base --is-ancestor $diagnosticSource HEAD
if ($LASTEXITCODE -ne 0) { throw 'HEAD does not contain the pinned diagnostic commit.' }
$extraFiles = @(& git -C $repo diff --name-only $diagnosticSource HEAD)
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify the effective source delta.' }
if (@($extraFiles | Where-Object { $_ -ne $scriptPath }).Count -ne 0) {
    throw 'HEAD differs from the diagnostic source beyond this build script.'
}
$dirty = @(& git -C $repo status --porcelain --untracked-files=no)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) {
    throw 'Tracked files must be clean before building.'
}
if ($Architecture -eq 'x64' -and
    [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITECTURE') -ne 'AMD64') {
    throw 'Use an x64 host process for an x64 target; ARM64 cross-targeting is supported on x64.'
}

$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
$binaryDirectory = Join-Path $BuildDirectory 'Release'
$ctest = Join-Path (Split-Path $CMake) 'ctest.exe'
$configureArguments = @(
    (Join-Path $repo 'tools\ci_build\build.py'),
    '--config', 'Release', '--build_dir', $BuildDirectory,
    '--skip_submodule_sync', '--skip_pip_install',
    '--cmake_path', $CMake, '--ctest_path', $ctest,
    '--cmake_generator', $Generator,
    '--parallel', "$Jobs", '--nvcc_threads', '1', '--flash_nvcc_threads', '1',
    '--use_vcpkg', '--use_vcpkg_ms_internal_asset_cache',
    '--update', '--use_cuda', '--cuda_home', $CudaHome, '--cudnn_home', $CudnnHome,
    '--skip_tests'
)
if ($Architecture -eq 'ARM64') { $configureArguments += '--arm64' }
$configureArguments += @(
    '--cmake_extra_defines',
    'CMAKE_CUDA_ARCHITECTURES=120-real;121-real',
    'CMAKE_CUDA_FLAGS=-allow-unsupported-compiler -Xcudafe --diag_suppress=2803',
    'onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON',
    'onnxruntime_BUILD_UNIT_TESTS=OFF',
    'onnxruntime_USE_FPA_INTB_GEMM=ON',
    'onnxruntime_PLUGIN_EP_VERSION=0.4.0-dev+64c1a75'
)
$buildArguments = @(
    '--build', $binaryDirectory, '--config', 'Release',
    '--target', 'onnxruntime_providers_cuda_plugin',
    '--', "/m:$Jobs", "/p:CL_MPCount=$Jobs", '/nodeReuse:false'
)
Write-Host "HEAD=$source diagnostic_source=$diagnosticSource provider_base=$providerBase"
Write-Host "Target=$Architecture; generator=$Generator; jobs=$Jobs; NVCC/Flash threads=1"
Write-Host 'Variant: installed CUDA/cuDNN; no BinSkim/Spectre; cudafe 2803 suppressed.'
Write-Host 'Diagnostics: ORT_CUDA_GEMM_AUTO_TUNE_DIAGNOSTICS=1; marker [CUDA GEMM diagnostic]'
if ($Mode -eq 'Plan') {
    Write-Output (@{ executable = $Python; arguments = $configureArguments } | ConvertTo-Json -Depth 3)
    Write-Output (@{ executable = $CMake; arguments = $buildArguments } | ConvertTo-Json -Depth 3)
    return
}
foreach ($path in @($Python, $CMake, $ctest, $CudaHome, $CudnnHome)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required input does not exist: $path" }
}
$cache = Join-Path $binaryDirectory 'CMakeCache.txt'
if ($Mode -ne 'Build' -and (Test-Path -LiteralPath $binaryDirectory)) {
    throw 'Configure requires a fresh build tree. Use -Mode Build to resume.'
}
if ($Mode -eq 'Build' -and -not (Test-Path -LiteralPath $cache)) {
    throw 'No configured build tree exists. Configure a fresh directory first.'
}
New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null
$recordPath = Join-Path $BuildDirectory 'diagnostic-build-inputs.json'
$record = [ordered]@{
    source = $source; diagnosticSource = $diagnosticSource; providerBase = $providerBase
    architecture = $Architecture; python = $Python; cmake = $CMake
    cudaHome = $CudaHome; cudnnHome = $CudnnHome; generator = $Generator
    configureArguments = $configureArguments; buildArguments = $buildArguments
}
if ($Mode -eq 'Build') {
    $previous = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
    foreach ($key in @('source', 'architecture', 'python', 'cmake', 'cudaHome', 'cudnnHome', 'generator')) {
        if ($previous.$key -ne $record[$key]) { throw "Resume input mismatch: $key" }
    }
}

function Invoke-LoggedCommand([string]$Executable, [string[]]$Arguments, [string]$Log) {
    Write-Host "Running $Executable; output: $Log"
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments *> $Log
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0) { throw "Command failed with exit code $exitCode. Preserved log: $Log" }
}

Push-Location $repo
try {
    if ($Mode -ne 'Build') {
        $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $recordPath -Encoding UTF8
        Invoke-LoggedCommand $Python $configureArguments (Join-Path $BuildDirectory 'configure.log')
    }
    if ($Mode -eq 'Configure') { return }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Invoke-LoggedCommand $CMake $buildArguments (Join-Path $BuildDirectory "build-$stamp.log")
    $dll = Join-Path $binaryDirectory 'Release\onnxruntime_providers_cuda.dll'
    $stream = [IO.File]::OpenRead($dll)
    $reader = [IO.BinaryReader]::new($stream)
    try {
        if ($reader.ReadUInt16() -ne 0x5A4D) { throw 'Output is not an MZ image.' }
        $stream.Position = 0x3C
        $peOffset = $reader.ReadInt32()
        $stream.Position = $peOffset
        if ($reader.ReadUInt32() -ne 0x4550) { throw 'Output has no PE signature.' }
        $machine = $reader.ReadUInt16()
    } finally {
        $reader.Dispose()
    }
    $expectedMachine = if ($Architecture -eq 'ARM64') { 0xAA64 } else { 0x8664 }
    if ($machine -ne $expectedMachine) { throw ('Unexpected PE machine: {0:X4}' -f $machine) }
    $result = [ordered]@{
        source = $source; diagnosticSource = $diagnosticSource; providerBase = $providerBase
        dll = $dll; sha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
        peMachine = ('{0:X4}' -f $machine)
        fileVersion = (Get-Item -LiteralPath $dll).VersionInfo.FileVersion
    }
    $result | ConvertTo-Json | Tee-Object -FilePath (Join-Path $BuildDirectory 'artifact.json')
} finally {
    Pop-Location
}
