# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


function Enter-MsvcEnv() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetArch
    )

    switch ($TargetArch) {
        "arm64" { $MsvcArch = "arm64" }
        "x86_64" { $MsvcArch = "amd64"}
        default { throw "Unknown target arch $TargetArch." }
    }

    & "$env:ProgramW6432\Microsoft Visual Studio\2022\Professional\Common7\Tools\Launch-VsDevShell.ps1" `
        -Arch $MsvcArch `
        -HostArch amd64 `
        -SkipAutomaticLocation

    if (-not $?) {
        throw "Could not activate MSVC environment for target arch $TargetArch"
    }
}

function Enter-PyVenv() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyVEnv
    )

    if ($env:VIRTUAL_ENV) {
        $LoadedVenv = (Resolve-Path "$env:VIRTUAL_ENV\").Path
        $DesiredVenv = (Resolve-Path "$PyVEnv\").Path

        if ($LoadedVenv -ne $DesiredVenv) {
            throw "Refusing to activate different Python venv."
        }
        else {
            Write-Host "Python venv $PyVEnv already activated."
        }
    }
    else {
        Write-Host "Activating Python venv $PyVEnv."
        . (Join-Path $PyVEnv "Scripts\Activate.ps1")
    }
}

function Get-DefaultCMakeGenerator() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$TargetPlatform,
        [Parameter(Mandatory = $true)]
        [string]$Arch
    )
    switch ($TargetPlatform) {
        "android" {
            "Ninja"
        }
        "windows" {
            $HostArch = (Get-HostArch)
            if ($Arch -eq $HostArch) {
                "Ninja"
            } else {
                Write-Host "Cross compiling for $Arch on $HostArch host. Cannot use Ninja."
                "Visual Studio 17 2022"
            }
        }
        default {
            throw "Unknown target platform $TargetPlatform."
        }
    }
}

function Get-HostArch() {
    switch ((Get-CimInstance Win32_operatingsystem).OSArchitecture) {
        "ARM 64-bit Processor" { "arm64" }
        "64-bit" { "x86_64" }
        default { throw "Unknown OS Architecture $OsArch." }
    }
}

function Get-QairtSdkFilePath() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )
    "$BuildDir\qairt-sdk-path-$Config.txt"
}

function Save-QairtSdkFilePath() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )

    $SdkFilePath = (Get-QairtSdkFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path "$SdkFilePath\..")) {
        New-Item -Path "$SdkFilePath\.." -ItemType Directory | Out-Null
    }
    $QairtSdkRoot | Out-File -FilePath $SdkFilePath
}

function Test-QairtSdkDiffers() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )

    $QairtSdkPathPath = (Get-QairtSdkFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path -Path $QairtSdkPathPath)) {
        return $True
    }

    $LastSdkPath = Get-Content -Path $QairtSdkPathPath
    return $LastSdkPath -ne $QairtSdkRoot
}

function Test-UpdateNeeded() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$TargetPlatform,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$CMakeGenerator,
        [Parameter(Mandatory = $true)]
        [bool]$Update
    )

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

    if (Test-QairtSdkDiffers -BuildDir $BuildDir -Config $Config) {
        Write-Host "Previous build used a different QAIRT SDK."
        return $True
    }

    Write-Host "No need to update build system."
    return $False
}
