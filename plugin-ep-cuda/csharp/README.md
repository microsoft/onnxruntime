# CUDA Plugin EP — NuGet Packaging

This directory contains the C# NuGet package project and test app for the CUDA plugin Execution Provider.

## Directory Structure

```
csharp/
├── pack_nuget.py                                   # Helper script to build the NuGet package
├── Microsoft.ML.OnnxRuntime.EP.Cuda/
│   ├── Microsoft.ML.OnnxRuntime.EP.Cuda.csproj     # NuGet package project (netstandard2.0)
│   ├── CudaEp.cs                                   # Helper class for native library resolution
│   └── README.md                                   # Package readme (shipped inside .nupkg)
└── test/
    └── CudaEpNuGetTest/
        ├── CudaEpNuGetTest.csproj                  # Test console app (net8.0)
        ├── Program.cs                              # Registers EP, runs inference, validates output
        ├── mul.onnx                                # Test model (element-wise multiply)
        └── generate_mul_model.py                   # Script to regenerate mul.onnx
```

## Prerequisites

- .NET SDK 8.0 or later
- A built CUDA plugin EP shared library
- NVIDIA GPU with CUDA support
- CUDA toolkit and cuDNN installed on the system

## Building the NuGet Package

Use `pack_nuget.py` to stage native binaries and run `dotnet pack`. The script copies everything into a staging
directory before building — the source tree is never modified. By default, an auto-cleaned temporary directory is used;
pass `--staging-dir` to use an explicit one (required when running with `--build-only` or `--pack-only`).

At least one binary directory (or `--artifacts-dir` with matching subdirectories) must be provided. Platforms without
a binary directory are skipped. Run `python pack_nuget.py --help` for the full list of options and their defaults.

### Pack with a local build (single platform)

```powershell
cd plugin-ep-cuda/csharp

python pack_nuget.py --version 0.1.0-dev `
  --binary-dir-win-x64 <path-to-win-x64-binaries>
```

### Pack multiple platforms

Each `--binary-dir-*` points at the directory containing that platform's already-built native binaries. In practice
the binaries are produced on different machines and combined in CI; locally you'd typically only set the one(s)
you have available.

```powershell
python pack_nuget.py --version 0.1.0-dev `
  --binary-dir-win-x64 <path-to-win-x64-binaries> `
  --binary-dir-linux-x64 <path-to-linux-x64-binaries> `
  --binary-dir-linux-aarch64 <path-to-linux-aarch64-binaries>
```

## Versioning

The package version is supplied to `pack_nuget.py` via `--version`. In the packaging pipeline, the release or
pre-release version is derived from [`VERSION_NUMBER`](../../VERSION_NUMBER).

## Inspecting the Package

The `.nupkg` is a ZIP file. To verify its contents:

```powershell
Expand-Archive nuget_output/Microsoft.ML.OnnxRuntime.EP.Cuda.0.1.0-dev.nupkg `
  -DestinationPath nuget_output/inspect -Force

Get-ChildItem nuget_output/inspect -Recurse | Select-Object FullName
```

Expected layout inside the package:

```
lib/netstandard2.0/Microsoft.ML.OnnxRuntime.EP.Cuda.dll
runtimes/win-x64/native/onnxruntime_providers_cuda_plugin.dll
runtimes/linux-x64/native/libonnxruntime_providers_cuda_plugin.so
runtimes/linux-arm64/native/libonnxruntime_providers_cuda_plugin.so
```

## Testing the Package

The test app registers the CUDA EP, creates a session, runs a simple Mul model, and validates the output.

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
"@ | Set-Content test/CudaEpNuGetTest/nuget.config

# Build and run
dotnet run --project test/CudaEpNuGetTest/CudaEpNuGetTest.csproj --configuration Release
```

A successful run prints `PASSED: All outputs match expected values.` and exits with code 0.

## Regenerating the Test Model

```bash
python test/CudaEpNuGetTest/generate_mul_model.py
```

Requires the `onnx` Python package.

## CI Pipeline

The NuGet packaging is integrated into the CUDA plugin pipeline:

- **Pipeline:** `tools/ci_build/github/azure-pipelines/plugin-cuda-pipeline.yml`
- **Packaging stage:** `tools/ci_build/github/azure-pipelines/stages/plugin-cuda-nuget-packaging-stage.yml`

The CI stage downloads build artifacts from all enabled platform stages, invokes `pack_nuget.py`, ESRP-signs the
package, and runs the test app on a GPU agent.

## Native Binaries Per Platform

| RID | Required Files |
|---|---|
| `win-x64` | `onnxruntime_providers_cuda_plugin.dll` |
| `linux-x64` | `libonnxruntime_providers_cuda_plugin.so` |
| `linux-arm64` | `libonnxruntime_providers_cuda_plugin.so` |
