<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <PackageReference Include="Microsoft.AI.MachineLearning" Version="[PackageVersion]" targetFramework="native" />
    <PackageReference Include="Microsoft.Windows.CppWinRT" Version="2.0.200615.7" targetFramework="native" />
  </ItemGroup>

  <PropertyGroup>
    <WindowsAI-Platform Condition="'$(Platform)' == 'Win32'">x86</WindowsAI-Platform>
    <WindowsAI-Platform Condition="'$(Platform)' != 'Win32'">$(Platform)</WindowsAI-Platform>
  </PropertyGroup>
  
  <PropertyGroup>
    <WindowsAIBinary>$(MSBuildThisFileDirectory)\packages\microsoft.ai.machinelearning\[PackageVersion]\runtimes\win-$(WindowsAI-Platform)\native\Microsoft.AI.MachineLearning.dll</WindowsAIBinary>
  </PropertyGroup>

  <ItemGroup>
    <Reference Include="$(MSBuildThisFileDirectory)\packages\microsoft.ai.machinelearning\[PackageVersion]\lib\uap10.0\Microsoft.AI.MachineLearning.winmd">
      <Implementation>$(WindowsAIBinary)</Implementation>
    </Reference>
  </ItemGroup>
</Project>
