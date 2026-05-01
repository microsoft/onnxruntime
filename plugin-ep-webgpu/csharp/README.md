# WebGPU Plugin EP — NuGet Packaging

This directory contains the C# NuGet package project and test app for the WebGPU plugin Execution Provider.

## Directory Structure

```
csharp/
├── pack_nuget.py                                   # Helper script to build the NuGet package
├── Microsoft.ML.OnnxRuntime.EP.WebGpu/
│   ├── Microsoft.ML.OnnxRuntime.EP.WebGpu.csproj   # NuGet package project (netstandard2.0)
│   ├── WebGpuEp.cs                                 # Helper class for native library resolution
│   └── README.md                                   # Package readme (shipped inside .nupkg)
└── test/
    └── WebGpuEpNuGetTest/
        ├── WebGpuEpNuGetTest.csproj                # Test console app (net8.0)
        ├── Program.cs                              # Registers EP, runs inference, validates output
        ├── nuget.config                            # NuGet source config (local feed for CI)
        ├── mul.onnx                                # Test model (element-wise multiply)
        └── generate_mul_model.py                   # Script to regenerate mul.onnx
```

## Prerequisites

- .NET SDK 8.0 or later
- A built WebGPU plugin EP binary (see below)

## Building the Plugin EP Binary

From the repo root:

```powershell
python tools/ci_build/build.py `
  --build_dir ./build/webgpu.plugin `
  --use_webgpu shared_lib `
  --use_vcpkg `
  --config Release `
  --parallel `
  --update --build
```

The plugin DLL will be at `build/webgpu.plugin/Release/Release/onnxruntime_providers_webgpu.dll`.

## Building the NuGet Package

Use `pack_nuget.py` to stage native binaries and run `dotnet pack`. The script copies
everything into a staging directory before building — the source tree is never modified.
By default, an auto-cleaned temporary directory is used; pass `--staging-dir` to use an
explicit one (required when running with `--build-only` or `--pack-only`).

### Pack with a local build (single platform)

```powershell
cd plugin-ep-webgpu/csharp

python pack_nuget.py --version 0.1.0-dev `
  --binary-dir-win-x64 ..\..\build\webgpu.plugin\Release\Release
```

### Pack multiple platforms

```powershell
python pack_nuget.py --version 0.1.0-dev `
  --binary-dir-win-x64 C:\builds\win_x64 `
  --binary-dir-win-arm64 C:\builds\win_arm64 `
  --binary-dir-linux-x64 /mnt/builds/linux_x64 `
  --binary-dir-macos-arm64 /mnt/builds/macos_arm64
```

### Script Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--version` | Yes | — | Package version string (e.g. `0.1.0`, `0.1.0-dev`) |
| `--output-dir` | No | `./nuget_output` | Directory for the `.nupkg` and `.snupkg` output |
| `--configuration` | No | `Release` | Build configuration |
| `--artifacts-dir` | No | — | CI mode: root directory with `win_x64/bin/`, `linux_x64/bin/`, etc. |
| `--binary-dir-win-x64` | No | — | Path to directory containing win-x64 binaries |
| `--binary-dir-win-arm64` | No | — | Path to directory containing win-arm64 binaries |
| `--binary-dir-linux-x64` | No | — | Path to directory containing linux-x64 binaries |
| `--binary-dir-macos-arm64` | No | — | Path to directory containing osx-arm64 (macOS arm64) binaries |
| `--staging-dir` | No | auto-cleaned temp dir | Explicit staging directory. Required with `--build-only` / `--pack-only`; caller owns its lifecycle (no auto-cleanup) |
| `--build-only` | No | `false` | Stage and build the managed DLL only; skip `dotnet pack`. Preserves the staging directory for a later `--pack-only` run |
| `--pack-only` | No | `false` | Skip staging/build and run `dotnet pack` against an existing staging directory (mutually exclusive with `--build-only`) |
| `--required-platforms` | No | — | Comma-separated list of platforms that MUST be staged successfully (CI-mode safety net) |

At least one binary directory (or `--artifacts-dir` with matching subdirectories) must be provided.
Platforms without a binary directory are skipped.

## Versioning

The package version is supplied to `pack_nuget.py` via `--version`. In the packaging pipeline, the release or
pre-release version is derived from [`plugin-ep-webgpu/VERSION_NUMBER`](../VERSION_NUMBER).

## Inspecting the Package

The `.nupkg` is a ZIP file. To verify its contents:

```powershell
Expand-Archive nuget_output/Microsoft.ML.OnnxRuntime.EP.WebGpu.0.1.0-dev.nupkg `
  -DestinationPath nuget_output/inspect -Force

Get-ChildItem nuget_output/inspect -Recurse | Select-Object FullName
```

Expected layout inside the package:

```
lib/netstandard2.0/Microsoft.ML.OnnxRuntime.EP.WebGpu.dll
runtimes/win-x64/native/onnxruntime_providers_webgpu.dll
runtimes/win-x64/native/dxil.dll
runtimes/win-x64/native/dxcompiler.dll
runtimes/win-arm64/native/...
runtimes/linux-x64/native/libonnxruntime_providers_webgpu.so
runtimes/osx-arm64/native/libonnxruntime_providers_webgpu.dylib
```

## Testing the Package

The test app registers the WebGPU EP, creates a session, runs a simple Mul model, and
validates the output. It requires a GPU with D3D12 or Vulkan support.

```powershell
# Point the test project's nuget.config at the pack output
$localFeed = (Resolve-Path nuget_output).Path
@"
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
    <add key="local" value="$localFeed" />
  </packageSources>
</configuration>
"@ | Set-Content test/WebGpuEpNuGetTest/nuget.config

# Build and run
dotnet run --project test/WebGpuEpNuGetTest/WebGpuEpNuGetTest.csproj --configuration Release
```

A successful run prints `PASSED: All outputs match expected values.` and exits with code 0.

## Regenerating the Test Model

```bash
python test/WebGpuEpNuGetTest/generate_mul_model.py
```

Requires the `onnx` Python package.

## CI Pipeline

The NuGet packaging is integrated into the WebGPU plugin pipeline:

- **Pipeline:** `tools/ci_build/github/azure-pipelines/plugin-webgpu-pipeline.yml`
- **Packaging stage:** `tools/ci_build/github/azure-pipelines/stages/plugin-webgpu-nuget-packaging-stage.yml`

The CI stage downloads build artifacts from all enabled platform stages, invokes
`pack_nuget.py`, ESRP-signs the package, and runs the test app on a GPU agent.

## Native Binaries Per Platform

| RID | Required Files |
|---|---|
| `win-x64` | `onnxruntime_providers_webgpu.dll`, `dxil.dll`, `dxcompiler.dll` |
| `win-arm64` | `onnxruntime_providers_webgpu.dll`, `dxil.dll`, `dxcompiler.dll` |
| `linux-x64` | `libonnxruntime_providers_webgpu.so` |
| `osx-arm64` | `libonnxruntime_providers_webgpu.dylib` |

On Windows, `dxil.dll` and `dxcompiler.dll` are the DirectX Shader Compiler binaries
downloaded from the [DXC GitHub releases](https://github.com/microsoft/DirectXShaderCompiler/releases).
The CI pipeline handles this automatically.
