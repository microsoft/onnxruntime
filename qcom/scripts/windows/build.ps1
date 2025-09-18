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
               HelpMessage = "What to do: build|test|generate_sln.")]
    [string]$Mode = "build",

    [Parameter(Mandatory = $false,
               HelpMessage = "The configuration to build.")]
    [string]$Config = "Release",

    [Parameter(Mandatory = $false,
               HelpMessage = "Force regeneration of build system.")]
    [bool]$Update = $false,

    [Parameter(Mandatory = $false,
               HelpMessage = "Path to a target-native python executable.")]
    [string]$TargetPyExe,

    [Parameter(Mandatory = $true,
               HelpMessage = "Python virtual environment to activate.")]
    [string]$PyVEnv
)

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\tools.ps1"
. "$RepoRoot\qcom\scripts\windows\utils.ps1"

$BuildRoot = (Join-Path $RepoRoot "build")

if ($Mode -eq "generate_sln") {
    $BuildDir = (Join-Path $BuildRoot "vs")
}
else {
    $BuildDir = (Join-Path $BuildRoot "windows-$Arch")
}

$ValidArchs = "aarch64", "arm64", "arm64ec", "x86_64"

if (-Not ($ValidArchs -contains $Arch)) {
    throw "Invalid arch $Arch. Supported architectures: $ValidArchs"
}

Enter-PyVenv $PyVEnv

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
    $CMakeGenerator = (Get-DefaultCMakeGenerator -Arch $Arch)

    if (Test-UpdateNeeded -BuildDir $BuildDir -Config $Config -CMakeGenerator $CMakeGenerator -Update $Update) {
        $BuildIsDirty = $true
        Save-QairtSdkFilePath -BuildDir $BuildDir -Config $Config
    } else {
        $BuildIsDirty = $false
    }
}

$ArchArgs = @()
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

$CommonArgs = `
    "--build_dir", $BuildDir, `
    "--build_shared_lib", `
    "--cmake_generator", $CmakeGenerator, `
    "--config", $Config, `
    "--parallel"

$QnnArgs = "--use_qnn", "--qnn_home", "$QairtSdkRoot"
$GenerateBuild = $false
$DoBuild = $false
$BuildWheel = $false
$MakeTestArchive = $false
$RunTests = $false
$TestRunner = "$RepoRoot\qcom\scripts\windows\run_tests.ps1"

if ($CMakeGenerator -eq "Ninja") {
    $CommonArgs += "--use_cache"
    $env:Path = "$(Get-CCacheBinDir);" + $env:Path
}

if ($TargetPyExe -ne "")
{
    # Wheels only supported when we can run Python for the target arch.
    $BuildWheel = $true
    $ArchArgs += "--enable_pybind"
    $BuildVEnv = (Join-Path $BuildDir "venv.build")
    Write-Host "Building Python wheel using $TargetPyExe"
}
else {
    $BuildVEnv = $PyVEnv
    Write-Host "Not building a Python wheel"
}

# The ORT build incorrectly enables use of Kleidiai when using Ninja on Windows,
# even if ArmNN is not requested. Manually turn it off.
$PlatformArgs = @("--no_kleidiai")

if ($CMakeGenerator -eq "Ninja") {
    # The default somehow gives us paths that are too long in CI
    $PlatformArgs += "--cmake_extra_defines", "CMAKE_OBJECT_PATH_MAX=240"
}

switch ($Mode) {
    "build" {
        if ($BuildIsDirty) {
            $GenerateBuild = $true
        }

        $DoBuild = $true
    }
    "generate_sln" {
        $GenerateBuild = $true
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

$CmakeBinDir = (Get-CMakeBinDir)
$env:Path = "$CmakeBinDir;" + $env:Path

Optimize-ToolsDir

Push-Location $RepoRoot

$failed = $false
if ($MakeTestArchive) {
    python.exe "$RepoRoot\qcom\scripts\all\archive_tests.py" `
        "--config=$Config" `
        "--qairt-sdk-root=$QairtSdkRoot" `
        "--target-platform=windows-$Arch"
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
        Copy-Item (Join-Path $CMakeBinDir "ctest.exe") -Destination (Join-Path $BuildDir $Config)
        Copy-Item -Path $RepoRoot\qcom\scripts\all\python_test_files.txt -Destination (Join-Path $BuildDir $Config)
    }

    if ($GenerateBuild -or $DoBuild) {
        try {
            python.exe "$RepoRoot\qcom\scripts\all\fetch_cmake_deps.py"

            if ($GenerateBuild) {
                if (-not (Test-Path $BuildVEnv)) {
                    & $TargetPyExe -m venv $BuildVEnv
                    if (-not $?) {
                        throw "Failed to create build virtual environment"
                    }
                }

                Invoke-PyVenv -PyVenv $BuildVEnv {
                    & python.exe -m pip install uv
                    & uv.exe pip install -r "$RepoRoot\tools\ci_build\github\windows\python\requirements.txt"
                    .\build.bat --update $ArchArgs $CommonArgs $QnnArgs $PlatformArgs
                    if (-not $?) {
                        throw "Failed to generate build"
                    }
                }
            }

            if ($DoBuild) {
                & cmake --build (Join-Path $BuildDir $Config) --config $Config
                if (-not $?) {
                    throw "Failed to build"
                }

                if ($BuildWheel) {
                    $BuildOutputDir = (Join-Path $BuildDir $Config)
                    if ($CmakeGenerator -eq "Visual Studio 17 2022") {
                        $BuildOutputDir = (Join-Path $BuildOutputDir $Config)
                    }

                    if ($env:ORT_NIGHTLY_BUILD) {
                        $PyNightlyArg = "--nightly_build"
                    }
                    Invoke-PyVenv -PyVenv $BuildVEnv {
                        Invoke-Directory -Path $BuildOutputDir {
                            & python.exe (Join-Path $RepoRoot "setup.py") `
                                bdist_wheel --wheel_name_suffix=qnn_qcom_internal $PyNightlyArg
                            if (-not $?) {
                                throw "Failed to build wheel"
                            }
                        }
                    }
                }
            }
        }
        finally {
            # Whatever happens, blow away mirror to avoid it showing up in git; it's okay, it's
            # very cheap to regenerate.
            Remove-Item -Recurse -Force (Join-Path $RepoRoot "mirror")
        }
    }

    if ($RunTests) {

        Push-Location (Join-Path $BuildDir $Config)
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
