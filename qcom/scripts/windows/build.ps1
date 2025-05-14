# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param (
    [Parameter(Mandatory = $true,
               HelpMessage = "The architecture for which to build.")]
    [string]$Arch,

    [Parameter(Mandatory = $false,
               HelpMessage = "Path to QAIRT SDK.")]
    [string]$QairtSdkRoot,

    [Parameter(Mandatory = $false,
               HelpMessage = "What to do: build|test.")]
    [string]$Mode = "Build",

    [Parameter(Mandatory = $false,
               HelpMessage = "The configuration to build.")]
    [string]$Config = "RelWithDebInfo",

    [Parameter(Mandatory = $false,
               HelpMessage = "Force regeneration of build system.")]
    [bool]$Update = $false,

    [Parameter(Mandatory = $false,
               HelpMessage = "Python virtual environment to activate.")]
    [string]$PyVEnv
)

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\tools.ps1"

$BuildDir = (Join-Path $RepoRoot "build\Windows-$Arch")
$CMakeGenerator = "Visual Studio 17 2022"
$ProtocPath = (Join-Path (Join-Path $BuildDir $Config) "Google.Protobuf.Tools.3.21.12\tools\windows_x64\protoc.exe")
$ValidArchs = "arm64", "arm64ec", "x86_64"

if ($PyVEnv -ne "") {
    . (Join-Path $PyVEnv "Scripts\Activate.ps1")
}

if ($QairtSdkRoot -eq "") {
    $QairtSdkRoot = (Install-Package qairt)
}
else {
    $QairtSdkRoot = Resolve-Path -Path $QairtSdkRoot
}

function Get-QairtSdkFilePath() {
    "$BuildDir\$Config\qairt-sdk-path.txt"
}

function Save-QairtSdkFilePath() {
    $QairtSdkRoot | Out-File -FilePath $(Get-QairtSdkFilePath)
}

function Test-QairtSdkDiffers() {
    $QairtSdkPathPath = Get-QairtSdkFilePath
    if (-Not (Test-Path -Path $QairtSdkPathPath)) {
        return $True
    }

    $LastSdkPath = Get-Content -Path $QairtSdkPathPath
    return $LastSdkPath -ne $QairtSdkRoot
}
function Test-UpdateNeeded() {
    if ($Update) {
        Write-Host "Build system update was requested."
        return $True
    }

    $SlnPath = "$BuildDir\$Config\onnxruntime.sln"
    if (-Not (Test-Path -Path $SlnPath)) {
        Write-Host "VS Solution $SlnPath does not exist."
        return $True
    }

    if (Test-QairtSdkDiffers) {
        Write-Host "Previous build used a different QAIRT SDK."
        return $True
    }

    Write-Host "No need to update build system."
    return $False
}

if (-Not ($ValidArchs -contains $Arch)) {
    throw "Invalid arch $Arch. Supported architectures: $ValidArchs"
}

$ArchArg = $null
if ($Arch -eq "x86_64")
{
    $HostArch = [System.Runtime.InteropServices.RuntimeInformation,mscorlib]::OSArchitecture
    if ($HostArch -ne "x64") {
        throw "Cross-compilation to $Arch is not supported on $HostArch host."
    }
} else {
    $ArchArg = "--$Arch"
}

$Actions = @()
$QnnArgs = "--use_qnn", "--qnn_home", "$QairtSdkRoot"

if ($Mode -eq "build")
{
    if (!(Test-Path $ProtocPath)) {
        Write-Host "$ProtocPath does not exist"
        $Nuget = (Join-Path (Install-Package nuget_win) "nuget.exe")
        & $Nuget `
            restore "$RepoRoot\packages.config" `
            -PackagesDirectory "$BuildDir\$Config" `
            -ConfigFile "$RepoRoot\NuGet.config"
    }

    if (Test-UpdateNeeded) {
        $Actions += "--update"
        Save-QairtSdkFilePath
    }

    $Actions += "--build"
}
elseif ($Mode -eq "test") {
    $Actions += "--test"
    if ($Arch -ne "arm64") {
        Write-Host "Disabling QNN tests on $Arch."
        $QnnArgs = @()
    }
}
else {
    throw "Unknown build mode $Mode."
}

Push-Location $RepoRoot
.\build.bat `
    $Actions `
    $ArchArg `
    --config "$Config" `
    --build_shared_lib `
    --parallel `
    --cmake_generator "$CmakeGenerator" `
    $QnnArgs `
    --build_dir "$BuildDir" `
    --path_to_protoc "$ProtocPath"
Pop-Location
