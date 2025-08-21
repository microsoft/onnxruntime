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
               HelpMessage = "What to do: build|test|generate_sln.")]
    [string]$Mode = "build",

    [Parameter(Mandatory = $false,
               HelpMessage = "The configuration to build.")]
    [string]$Config = "Release",

    [Parameter(Mandatory = $false,
               HelpMessage = "Force regeneration of build system.")]
    [bool]$Update = $false,

    [Parameter(Mandatory = $false,
               HelpMessage = "Python virtual environment to activate.")]
    [string]$PyVEnv
)

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\tools.ps1"
. "$RepoRoot\qcom\scripts\windows\utils.ps1"

$BuildRoot = (Join-Path $RepoRoot "build")
if ($TargetPlatform -eq "android") {
    $TargetPlatformArch = $TargetPlatform
} else {
    $TargetPlatformArch = "$TargetPlatform-$Arch"
}

if ($Mode -eq "generate_sln") {
    $BuildDir = (Join-Path $BuildRoot "vs")
}
else {
    $BuildDir = (Join-Path $BuildRoot "$TargetPlatformArch")
}

$ValidArchs = "arm64", "arm64ec", "x86_64"

if (-Not ($ValidArchs -contains $Arch)) {
    throw "Invalid arch $Arch. Supported architectures: $ValidArchs"
}

if ($PyVEnv -ne "") {
    Enter-PyVenv $PyVEnv
}

if ($QairtSdkRoot -eq "") {
    $QairtSdkRoot = (Get-QairtRoot)
}
else {
    $QairtSdkRoot = Resolve-Path -Path $QairtSdkRoot
}

if ($Mode -eq "generate_sln") {
    $CMakeGenerator = "Visual Studio 17 2022"
    $BuildIsDirty = $true
}
else {
    $CMakeGenerator = (Get-DefaultCMakeGenerator -TargetPlatform $TargetPlatform -Arch $Arch)

    if (Test-UpdateNeeded -BuildDir $BuildDir -TargetPlatform $TargetPlatform -Config $Config -CMakeGenerator $CMakeGenerator -Update $Update) {
        $BuildIsDirty = $true
        Save-QairtSdkFilePath -BuildDir $BuildDir -Config $Config
    } else {
        $BuildIsDirty = $false
    }
}

$ArchArgs = @()
$EpBuildDir = $BuildDir
if ($TargetPlatform -eq "windows") {
    if ($CMakeGenerator -eq "Ninja") {
        # We don't have Visual Studio to set up the build environment so do it
        # manually with somthing akin to vcvarsall.bat.
        Enter-MsvcEnv -TargetArch $Arch
    } elseif ($Arch -ne "x86_64") {
        # Tell the EP build that we're cross-compiling to ARM64.
        # We do not do this when using Ninja because our fake vcvars handles
        # cross-compilation flags.
        $ArchArgs += "--$Arch"
    }

    if ($Arch -eq (Get-HostArch) -and $Arch -eq "x86_64")
    {
        # Wheels only supported when not cross-compiling
        # Currently disabled on ARM64 until we can confirm python is native.
        $ArchArgs += "--build_wheel"
    }
}

$CommonArgs = `
    "--build_dir", $EpBuildDir, `
    "--build_shared_lib", `
    "--cmake_generator", $CmakeGenerator, `
    "--config", $Config, `
    "--parallel", `
    "--wheel_name_suffix", "qcom-internal", `
    "--compile_no_warning_as_error"

$Actions = @()
$QnnArgs = "--use_qnn", "--qnn_home", "$QairtSdkRoot"
$MakeTestArchive = $false
$RunTests = $false
$TestRunner = $null

if ($CMakeGenerator -eq "Ninja") {
    $CommonArgs += "--use_cache"
    $env:Path = "$(Get-CCacheBinDir);" + $env:Path
}

switch ($TargetPlatform) {
    "windows" {
        $TestRunner = "$RepoRoot\qcom\scripts\windows\run_tests.ps1"

        # The ORT build incorrectly enables use of Kleidiai when using Ninja on Windows,
        # even if ArmNN is not requested. Manually turn it off.
        $PlatformArgs = $ArchArg, "--no_kleidiai"

        if ($CMakeGenerator -eq "Ninja") {
            # The default somehow gives us paths that are too long in CI
            $PlatformArgs += "--cmake_extra_defines", "CMAKE_OBJECT_PATH_MAX=240"
        }

        switch ($Mode) {
            "build" {
                if ($BuildIsDirty) {
                    $Actions += "--update"
                }

                $Actions += "--build"
            }
            "generate_sln" {
                $Actions += "--update"
            }
            "test" {
                $RunTests = $true
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

        $env:Path = "$(Get-JavaBinDir);" + $env:Path

        if ($null -ne $env:ANDROID_HOME -and $null -ne $env:ANDROID_NDK_HOME) {
            $AndroidSdkPath = $env:ANDROID_HOME
            $AndroidNdkPath = $env:ANDROID_NDK_HOME
        }
        else {
            $AndroidSdkPath = (Get-AndroidSdkRoot)
            $AndroidNdkPath = (Get-AndroidNdkRoot)
        }

        $QnnArgs = "--use_qnn", "static_lib", "--qnn_home", "$QairtSdkRoot"
        $PlatformArgs = `
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

$CmakeBinDir = (Get-CMakeBinDir)
$env:Path = "$CmakeBinDir;" + $env:Path

Optimize-ToolsDir

Push-Location $RepoRoot

$failed = $false
if ($MakeTestArchive) {
    python.exe "$RepoRoot\qcom\scripts\all\archive_tests.py" `
        "--config=$Config" `
        "--qairt-sdk-root=$QairtSdkRoot" `
        "--target-platform=$TargetPlatformArch"
    if (-not $?) {
        $failed = $true
    }
}
else {
    if ($CMakeGenerator -eq "Ninja") {
        $env:Path = "$(Get-NinjaBinDir);" + $env:Path
    }

    # This platform supports running tests on the host. Prep the build directory
    # to run with our ctest wrapper
    if ($TestRunner) {
        if (-not (Test-Path (Join-Path $BuildDir $Config))) {
            New-Item -ItemType Directory (Join-Path $BuildDir $Config) | Out-Null
        }
        Copy-Item -Path $TestRunner -Destination (Join-Path $BuildDir $Config)
        Copy-Item "$CMakeBinDir\ctest.exe" -Destination (Join-Path $BuildDir $Config)
    }

    if ($Actions.Count -gt 0) {
        try {
            python.exe "$RepoRoot\qcom\scripts\all\fetch_cmake_deps.py"

            .\build.bat `
                $Actions `
                $ArchArgs `
                $CommonArgs `
                $QnnArgs `
                $PlatformArgs

            if (-not $?) {
                $failed = $true
            }
        }
        finally {
            # Whatever happens, blow away mirror to avoid it showing up in git; it's okay, it's
            # very cheap to regenerate.
            Remove-Item -Recurse -Force "$RepoRoot\mirror"
        }
    }

    if ($RunTests) {

        Push-Location "$BuildDir\$Config"
        $OnnxModelsRoot = (Get-OnnxModelsRoot)
        & .\run_tests.ps1 -OnnxModelsRoot $OnnxModelsRoot

        if (-not $?) {
            $failed = $true
        }
    }
}

if ($failed) {
    throw "Build failure"
}

Pop-Location
