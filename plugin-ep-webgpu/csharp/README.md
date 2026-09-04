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
        └── generate_mul_model.py                   # Generates mul.onnx (the test model) on demand
```

## Prerequisites

- .NET SDK 8.0 or later
- Pre-built WebGPU plugin EP binaries (from CI or a local build)

## Building the NuGet Package

Use `pack_nuget.py` to stage native binaries and run `dotnet pack`. The script copies everything into a staging
directory before building — the source tree is never modified. By default, an auto-cleaned temporary directory is used;
pass `--staging-dir` to use an explicit one (required when running with `--build-only` or `--pack-only`).

At least one binary directory (or `--artifacts-dir` with matching subdirectories) must be provided. Platforms without
a binary directory are skipped. Run `python pack_nuget.py --help` for the full list of options and their defaults.

### Pack with a local build (single platform)

```powershell
cd plugin-ep-webgpu/csharp

python pack_nuget.py --version 0.1.0-dev `
  --binary-dir-win-x64 <path-to-win-x64-binaries>
```

### Pack multiple platforms

Each `--binary-dir-*` points at the directory containing that platform's already-built native binaries. In practice
the four binaries are produced on different machines and combined in CI; locally you'd typically only set the one(s)
you have available.

```powershell
python pack_nuget.py --version 0.1.0-dev `
  --binary-dir-win-x64 <path-to-win-x64-binaries> `
  --binary-dir-win-arm64 <path-to-win-arm64-binaries> `
  --binary-dir-linux-x64 <path-to-linux-x64-binaries> `
  --binary-dir-macos-arm64 <path-to-macos-arm64-binaries>
```

### Pack from downloaded CI artifacts

When the per-platform binaries have been downloaded as pipeline artifacts into a single root directory (with one
subdirectory per platform), use `--artifacts-dir` instead of the individual `--binary-dir-*` flags:

```powershell
python pack_nuget.py --version 0.1.0-dev --artifacts-dir <path-to-artifacts-root>
```

## Testing

The test app registers the WebGPU EP, creates a session, runs a simple Mul model, and validates the output.

First generate the test model if you haven't already (see [Regenerating the Test Model](#regenerating-the-test-model)),
then point the test project's `nuget.config` at pack_nuget.py's output and run the test.

> **Note:** the snippet below **overwrites** `test/WebGpuEpNuGetTest/nuget.config` if it already exists. Back up
> any local changes first.

```powershell
# Point the test project's nuget.config at pack_nuget.py's output
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

## Versioning

The package version is supplied to `pack_nuget.py` via `--version`. In the packaging pipeline, that version is
derived from [`../VERSION_NUMBER`](../VERSION_NUMBER) (see
[`set-plugin-ep-build-variables-step.yml`](../../tools/ci_build/github/azure-pipelines/templates/set-plugin-ep-build-variables-step.yml)).
