# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param (
    [Parameter(Mandatory = $true,
               HelpMessage = "The architecture for which to build.")]
    [string]$Arch,

    [Parameter(Mandatory = $true,
               HelpMessage = "The platform for which to build.")]
    [string]$TargetPlatform,

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

$BuildRoot = (Join-Path $RepoRoot "build")
if ($TargetPlatform -eq "android") {
    $TargetPlatformArch = $TargetPlatform
    $CMakeGenerator = "Ninja"
} else {
    $TargetPlatformArch = "$TargetPlatform-$Arch"
    $CMakeGenerator = "Visual Studio 17 2022"
}
$BuildDir = (Join-Path $BuildRoot "$TargetPlatformArch")
$ProtocPath = (Join-Path (Join-Path $BuildDir $Config) "Google.Protobuf.Tools.3.21.12\tools\windows_x64\protoc.exe")
$ValidArchs = "arm64", "arm64ec", "x86_64"

if ($PyVEnv -ne "") {
    . (Join-Path $PyVEnv "Scripts\Activate.ps1")
}

if ($QairtSdkRoot -eq "") {
    $QairtSdkRoot = (Get-PackageContentDir qairt)
}
else {
    $QairtSdkRoot = Resolve-Path -Path $QairtSdkRoot
}

function Get-QairtSdkFilePath() {
    "$BuildDir\qairt-sdk-path-$Config.txt"
}

function Save-QairtSdkFilePath() {
    $SdkFilePath = (Get-QairtSdkFilePath)
    if (-Not (Test-Path "$SdkFilePath\..")) {
        New-Item -Path "$SdkFilePath\.." -ItemType Directory | Out-Null
    }
    $QairtSdkRoot | Out-File -FilePath $SdkFilePath
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

    if ($CMakeGenerator -eq "Ninja") {
        $BuildNinjaPath = "$BuildDir\$Config\build.ninja"
        if (-Not (Test-Path -Path $BuildNinjaPath)) {
            Write-Host "$BuildNinjaPath does not exist."
            return $True
        }
    } else {
        $SlnPath = "$BuildDir\$Config\onnxruntime.sln"
        if (-Not (Test-Path -Path $SlnPath)) {
            Write-Host "VS Solution $SlnPath does not exist."
            return $True
        }
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

if (Test-UpdateNeeded) {
    $BuildIsDirty = $true
    Save-QairtSdkFilePath
} else {
    $BuildIsDirty = $false
}

$CommonArgs = `
    "--build_dir", $BuildDir, `
    "--build_shared_lib", `
    "--cmake_generator", $CmakeGenerator, `
    "--config", $Config, `
    "--parallel", `
    "--path_to_protoc", $ProtocPath

$Actions = @()
$QnnArgs = "--use_qnn", "--qnn_home", "$QairtSdkRoot"
$MakeTestArchive = $false

switch ($TargetPlatform) {
    "windows" {
        switch ($Mode) {
            "build" {
                if ($BuildIsDirty) {
                    $Actions += "--update"
                }

                $Actions += "--build"
                $PlatformArgs = $ArchArg
            }
            "test" {
                $Actions += "--test"
                if ($Arch -ne "arm64") {
                    Write-Host "Disabling QNN tests on $Arch."
                    $QnnArgs = @()
                }
            }
            "archive" {
                $MakeTestArchive = $true
            }
            default {
                throw "Unknown build mode $Mode."
            }
        }
    }
    "android" {
        if ($BuildIsDirty -and (Test-Path "$BuildDir\$Config")) {
            # The ORT Android build doesn't seem to support --update, but our QNN root has changed
            # so we really want to re-run cmake. Blow away the build.
            Write-Host "Build is dirty: blowing away $BuildDir\$Config"
            Remove-Item -Recurse -Force "$BuildDir\$Config"
            if (-not $?) {
                throw "Failed to scrub $BuildDir\$Config"
            }
        }

        $env:Path = "$(Get-PackageBinDir java_windows_x86_64);" + $env:Path
        $env:Path = "$(Get-PackageBinDir ccache_windows_x86_64);" + $env:Path

        if ($null -ne $env:ANDROID_HOME -and $null -ne $env:ANDROID_NDK_HOME) {
            $AndroidSdkPath = $env:ANDROID_HOME
            $AndroidNdkPath = $env:ANDROID_NDK_HOME
        }
        else {
            $AndroidSdkPath = (Get-AndroidSdkRoot)
            $AndroidNdkPath = (Get-AndroidNdkRoot)
        }

        $QnnArgs = "--use_qnn", "static_lib", "--qnn_home", "$QairtSdkRoot"
        $PlatformArgs = "--use_cache", `
            "--android_sdk_path", $AndroidSdkPath, `
            "--android_ndk_path", $AndroidNdkPath, `
            "--android_abi", "arm64-v8a", `
            "--android_api", "27"

        switch ($Mode) {
            "build" {
                $Actions += "--android"
            }
            "test" {
                throw "-Mode test not supported with -TargetPlatform $TargetPlatform."
            }
            "archive" {
                $MakeTestArchive = $true
            }
            default {
                throw "Invalid mode '$Mode'."
            }
        }
    }
    default {
        throw "Unknown target platform $TargetPlatform."
    }
}

$CmakeBinDir = (Get-PackageBinDir cmake_windows_x86_64)
$env:Path = "$CmakeBinDir;" + $env:Path

Push-Location $RepoRoot

if ($MakeTestArchive) {
    python.exe "$RepoRoot\qcom\scripts\all\archive_tests.py" `
        "--cmake-bin-dir=$CmakeBinDir" `
        "--config=$Config" `
        "--qairt-sdk-root=$QairtSdkRoot" `
        "--target-platform=$TargetPlatformArch"
}
else {
    if (!(Test-Path $ProtocPath)) {
        Write-Host "$ProtocPath does not exist"
        $Nuget = (Join-Path (Get-PackageBinDir nuget_win) "nuget.exe")
        & $Nuget `
            restore "$RepoRoot\packages.config" `
            -PackagesDirectory "$BuildDir\$Config" `
            -ConfigFile "$RepoRoot\NuGet.config"
    }

    if ($CMakeGenerator -eq "Ninja") {
        $env:Path = "$(Get-PackageBinDir ninja_windows_x86_64);" + $env:Path
    }

    .\build.bat `
        $Actions `
        $CommonArgs `
        $QnnArgs `
        $PlatformArgs
}

if (-not $?) {
    throw "Build failure"
}

Pop-Location
