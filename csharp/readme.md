# ORT C# Managed Library

The solution files here are used to produce nuget packages for the C# bindings.

Note that the project naming is currently confusing and needs updating. 

  - The Microsoft.ML.OnnxRuntime project produces the Microsoft.ML.OnnxRuntime.**Managed** nuget package.
  - The Microsoft.ML.OnnxRuntime nuget package contains the native (i.e. C++) code for various platforms.

## Solution files

The main solution file is OnnxRuntime.CSharp.sln. This includes desktop and Xamarin mobile projects.
OnnxRuntime.DesktopOnly.CSharp.sln is a copy of that with all the mobile projects removed. This is 
due to there being no way to selectively exclude a csproj from the sln if Xamarin isn't available. 

If changes are required, either update the main solution first and copy the relevant changes across,
or copy the entire file and remove the mobile projects (anything with iOS, Android or Droid in the name). 

## Development setup:

### Requirements:

NOTE: The usage of this solution is primarily for ORT developers creating the managed Microsoft.ML.OnnxRuntime.Managed 
      nuget package. Due to that, the requirements are quite specific.

Visual Studio 2022 v17.2.4 or later, with Xamarin workloads
  - v17.2.4 installs dotnet sdk 6.0.301
  - in theory you could use an earlier VS version and download dotnet SDK 6.0.300+ from https://dotnet.microsoft.com/en-us/download/dotnet/6.0
    - untested

There's no good way to use Visual Studio 2022 17.3 Preview in a CI, so we currently have to build pre-.net6 targets
using VS, and .net6 targets using dotnet. We can't build them all using dotnet as the xamarin targets require msbuild. 
We can't package them using dotnet as that also requires msbuild.

Once the official VS 2022 release supports .net6 and is available in the CI we can revert to the original simple 
setup of building everything using msbuild.

To test packaging locally you will also need nuget.exe. 
Download from https://www.nuget.org/downloads. 
Put in a folder (e.g. C:\Program Files (x86)\NuGet).
Add that folder to your PATH. 

### Magic incantations to build the nuget managed package locally:

If we're starting with VS 2022 17.2.4 we should have dotnet sdk 6.0.301

Make sure all the required workloads are installed
  `dotnet workload install android ios maccatalyst macos`
    - original example from [here](https://github.com/Sweekriti91/maui-samples/blob/swsat/devops/6.0/Apps/WeatherTwentyOne/devops/AzureDevOps/azdo_windows.yml): 
      - `dotnet workload install android ios maccatalyst macos maui --source https://aka.ms/dotnet6/nuget/index.json --source https://api.nuget.org/v3/index.json`
    - don't need 'maui' in this list until we update the sample/test apps
    - didn't seem to need --source arg/s for local build. YMMV.

Build pre-net6 targets
  `msbuild -t:restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=PreNet6`
  `msbuild -t:build .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=PreNet6`

  Need to run msbuild twice - once to restore which creates some json configs that are needed like 
  Microsoft.ML.OnnxRuntime\obj\project.assets.json, and once to build using the configs.

Build net6 targets
  `dotnet build .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=Net6`
  
  The dotnet build does the restore internally.

Create project.assets.json in obj dir with all targets so the nuget package creation includes them all
  `msbuild -t:restore .\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj -p:SelectedTargets=All`

Create nuget package
  `msbuild .\OnnxRuntime.CSharp.proj -t:CreatePackage -p:OrtPackageId=Microsoft.ML.OnnxRuntime -p:Configuration=Debug -p:Platform="Any CPU"`
