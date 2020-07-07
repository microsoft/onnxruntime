﻿<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <Import Project="packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.props" Condition="Exists('packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.props')" />
  <Import Project="$(MSBuildExtensionsPath)\$(MSBuildToolsVersion)\Microsoft.Common.props" Condition="Exists('$(MSBuildExtensionsPath)\$(MSBuildToolsVersion)\Microsoft.Common.props')" />
  <PropertyGroup>
    <Configuration Condition=" '$(Configuration)' == '' ">Debug</Configuration>
    <Platform Condition=" '$(Platform)' == '' ">AnyCPU</Platform>
    <ProjectGuid>{717C645E-9C4A-4EBF-A0AF-676B265E8CDA}</ProjectGuid>
    <OutputType>Exe</OutputType>
    <RootNamespace>Microsoft.AI.MachineLearning.Tests.CSharp</RootNamespace>
    <AssemblyName>Microsoft.AI.MachineLearning.Tests.CSharp</AssemblyName>
    <TargetFrameworkVersion>v4.7.2</TargetFrameworkVersion>
    <FileAlignment>512</FileAlignment>
    <AutoGenerateBindingRedirects>true</AutoGenerateBindingRedirects>
    <Deterministic>true</Deterministic>
    <NuGetPackageImportStamp>
    </NuGetPackageImportStamp>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)|$(Platform)' == 'Debug|AnyCPU' ">
    <PlatformTarget>AnyCPU</PlatformTarget>
    <DebugSymbols>true</DebugSymbols>
    <DebugType>full</DebugType>
    <Optimize>false</Optimize>
    <OutputPath>bin\Debug\</OutputPath>
    <DefineConstants>DEBUG;TRACE</DefineConstants>
    <ErrorReport>prompt</ErrorReport>
    <WarningLevel>4</WarningLevel>
  </PropertyGroup>
  <PropertyGroup Condition=" '$(Configuration)|$(Platform)' == 'Release|AnyCPU' ">
    <PlatformTarget>AnyCPU</PlatformTarget>
    <DebugType>pdbonly</DebugType>
    <Optimize>true</Optimize>
    <OutputPath>bin\Release\</OutputPath>
    <DefineConstants>TRACE</DefineConstants>
    <ErrorReport>prompt</ErrorReport>
    <WarningLevel>4</WarningLevel>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)' == 'Debug|x64'">
    <DebugSymbols>true</DebugSymbols>
    <OutputPath>bin\x64\Debug\</OutputPath>
    <DefineConstants>DEBUG;TRACE</DefineConstants>
    <DebugType>full</DebugType>
    <PlatformTarget>x64</PlatformTarget>
    <LangVersion>7.3</LangVersion>
    <ErrorReport>prompt</ErrorReport>
    <CodeAnalysisRuleSet>MinimumRecommendedRules.ruleset</CodeAnalysisRuleSet>
    <Prefer32Bit>true</Prefer32Bit>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)' == 'Release|x64'">
    <OutputPath>bin\x64\Release\</OutputPath>
    <DefineConstants>TRACE</DefineConstants>
    <Optimize>true</Optimize>
    <DebugType>pdbonly</DebugType>
    <PlatformTarget>x64</PlatformTarget>
    <LangVersion>7.3</LangVersion>
    <ErrorReport>prompt</ErrorReport>
    <CodeAnalysisRuleSet>MinimumRecommendedRules.ruleset</CodeAnalysisRuleSet>
    <Prefer32Bit>true</Prefer32Bit>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include="Microsoft.AI.MachineLearning.Interop, Version=1.0.0.0, Culture=neutral, processorArchitecture=MSIL">
      <HintPath>packages\Microsoft.AI.MachineLearning.[PackageVersion]\lib\netstandard2.0\Microsoft.AI.MachineLearning.Interop.dll</HintPath>
    </Reference>
    <Reference Include="Microsoft.Windows.SDK.NET, Version=10.0.18362.3, Culture=neutral, processorArchitecture=MSIL">
      <HintPath>packages\Microsoft.Windows.SDK.NET.10.0.18362.3-preview\lib\netstandard2.0\Microsoft.Windows.SDK.NET.dll</HintPath>
    </Reference>
    <Reference Include="System" />
    <Reference Include="System.Core" />
    <Reference Include="System.Xml.Linq" />
    <Reference Include="System.Data.DataSetExtensions" />
    <Reference Include="Microsoft.CSharp" />
    <Reference Include="System.Data" />
    <Reference Include="System.Net.Http" />
    <Reference Include="System.Xml" />
    <Reference Include="winrt.runtime, Version=0.1.0.1677, Culture=neutral, processorArchitecture=MSIL">
      <HintPath>packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\lib\netstandard2.0\winrt.runtime.dll</HintPath>
    </Reference>
  </ItemGroup>
  <ItemGroup>
    <Compile Include="main.cs" />
  </ItemGroup>
  <ItemGroup>
    <None Include="packages.config" />
  </ItemGroup>
  <ItemGroup>
    <None Include="..\..\testdata\squeezenet.onnx">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <Visible>true</Visible>
    </None>
  </ItemGroup>
  <ItemGroup>
    <None Include="..\..\..\winml\test\collateral\images\kitten_224.png">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <Visible>true</Visible>
    </None>
  </ItemGroup>
  <ItemGroup>
    <Folder Include="Properties\" />
  </ItemGroup>
  <Import Project="$(MSBuildToolsPath)\Microsoft.CSharp.targets" />
  <Target Name="EnsureNuGetPackageBuildImports" BeforeTargets="PrepareForBuild">
    <PropertyGroup>
      <ErrorText>This project references NuGet package(s) that are missing on this computer. Use NuGet Package Restore to download them.  For more information, see http://go.microsoft.com/fwlink/?LinkID=322105. The missing file is {0}.</ErrorText>
    </PropertyGroup>
    <Error Condition="!Exists('packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.props')" Text="$([System.String]::Format('$(ErrorText)', 'packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.props'))" />
    <Error Condition="!Exists('packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.targets')" Text="$([System.String]::Format('$(ErrorText)', 'packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.targets'))" />
    <Error Condition="!Exists('packages\Microsoft.AI.MachineLearning.[PackageVersion]\build\netstandard2.0\Microsoft.AI.MachineLearning.targets')" Text="$([System.String]::Format('$(ErrorText)', 'packages\Microsoft.AI.MachineLearning.[PackageVersion]\build\netstandard2.0\Microsoft.AI.MachineLearning.targets'))" />
  </Target>
  <Import Project="packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.targets" Condition="Exists('packages\Microsoft.Windows.CsWinRT.0.1.0-prerelease.200512.7\build\Microsoft.Windows.CsWinRT.targets')" />
  <Import Project="packages\Microsoft.AI.MachineLearning.[PackageVersion]\build\netstandard2.0\Microsoft.AI.MachineLearning.targets" Condition="Exists('packages\Microsoft.AI.MachineLearning.[PackageVersion]\build\netstandard2.0\Microsoft.AI.MachineLearning.targets')" />
</Project>