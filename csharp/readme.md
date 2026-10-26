# ORT C# Managed Library

The solution files here are used to produce nuget packages for the C# bindings.

Note that the project naming is currently confusing and needs updating.

  - The Microsoft.ML.OnnxRuntime project produces the Microsoft.ML.OnnxRuntime.**Managed** nuget package.
  - The Microsoft.ML.OnnxRuntime nuget package contains the native (i.e. C++) code for various platforms.

## Solution files

The main solution file is OnnxRuntime.CSharp.sln. This includes desktop and MAUI mobile projects.
OnnxRuntime.DesktopOnly.CSharp.sln excludes the mobile projects and can be built without mobile workloads.

If changes are required, either update the main solution first and copy the relevant changes across,
or copy the entire file and remove the mobile projects (anything with iOS, Android or Droid in the name).

## Development setup:

### Desktop CI tests

`Microsoft.ML.OnnxRuntime.Tests.NetCoreApp` requires the .NET 10 SDK and runs on the .NET 10 runtime.
After building the native libraries, run it from the repository root:

```powershell
dotnet test csharp\test\Microsoft.ML.OnnxRuntime.Tests.NetCoreApp\Microsoft.ML.OnnxRuntime.Tests.NetCoreApp.csproj -p:IncludeMobileTargets=false
```

The managed library and shared test library target `netstandard2.0;net10.0`. Older consumers can use the
`netstandard2.0` assembly; the framework-specific tensor APIs are provided by the `net10.0` assembly.
Package-validation tests, plugin test apps, samples, performance tools, and API documentation also use
.NET 10. Linux CI and packaging containers install SDK 10, which includes runtime 10, even when their
cached base image contains an older SDK.

MAUI projects and platform-specific managed package assets use .NET 10 workloads. Install the required
workloads with the .NET 10 SDK; mobile deployment still requires the corresponding Android SDK or Apple
toolchain.

### Requirements:

#### Windows

NOTE: The usage of this solution is primarily for ORT developers creating the managed Microsoft.ML.OnnxRuntime.Managed
      nuget package. Due to that, the requirements are quite specific.

Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).
Use `dotnet` or `dotnet msbuild` for managed builds so the SDK's MSBuild is used instead of an older
Visual Studio installation. Building .NET 10 projects in the Visual Studio IDE requires Visual Studio 2026
or later. This does not change the compiler requirements for ORT's native C++ code.

To test packaging locally you will also need nuget.exe.
Download from https://www.nuget.org/downloads.
Put in a folder (e.g. C:\Program Files (x86)\NuGet).
Add that folder to your PATH.

#### Linux

Install the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0).
Linux builds use the .NET CLI to build and package the managed assemblies.

### Magic incantations to build the nuget managed package locally:

#### Windows

The IncludeMobileTargets property determines whether the MAUI targets are included. This is set to 'true' for the
real package build.

Make sure all the required workloads are installed if the MAUI targets are included.
  `dotnet workload install maui android ios maccatalyst`

From the `csharp` directory, restore and build the managed package:
  `dotnet restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:IncludeMobileTargets=true`
  `dotnet build .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj --no-restore -p:IncludeMobileTargets=true`

Create nuget package
  `dotnet msbuild .\OnnxRuntime.CSharp.proj -t:CreatePackage -p:OrtPackageId=Microsoft.ML.OnnxRuntime -p:Configuration=Debug -p:Platform="Any CPU"`

#### Linux

For example, to build a CUDA GPU package, just run:
```bash
./build.sh \
  --config="Release" \
  --cmake_generator Ninja \
  --use_cuda \
    --cuda_home=/usr/local/cuda \
    --cudnn_home=/usr \
  --build_nuget \
  --msbuild_extra_options IncludeMobileTargets=false
```
For a CPU development package, omit the CUDA options. Linux builds exclude mobile targets.

A `.nupkg` file will be produced at you build root, say, `build/Release`.

To consume the package, in your .net project,
```bash
nuget add <path/to/packages.nuget> -Source ./packages/
dotnet add package microsoft.ml.onnxruntime.managed -s ./packages --prerelease
dotnet add package microsoft.ml.onnxruntime.gpu -s ./packages --prerelease
```
