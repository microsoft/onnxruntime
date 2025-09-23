# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT


function Assert-Success() {
    param(
        [scriptblock]$Code,
        [Parameter(Mandatory = $false)]
        [string]$ErrorMessage = "Execution failed"
    )
    Invoke-Command -ScriptBlock $Code

    # This has some limitations. In particular, not every command indicates error
    # by $LASTEXITCODE other than 0. This is especially true of built-in commands
    # such as New-Item, but also some native things like robocopy. Still, we choose
    # to use $LASTEXITCODE because Invoke-Command does not propogate the success
    # of the command it invoked.
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

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
            throw "Refusing to activate different Python venv ($LoadedVenv vs $DesiredVenv)."
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

function Exit-PyVenv() {
    if (-not $env:VIRTUAL_ENV) {
        throw "Cannot deactivate: no virtual environment is active."
    }

    # If we simply deactivate, any changes we've made to $env:PATH since we activated will be lost.

    # Figure out what $env:Path should be
    $PathNoVenv = (($env:Path.Split(";") | Where-Object { $_ -ne "${env:VIRTUAL_ENV}\Scripts"}) -join ";")

    deactivate

    $env:Path = $PathNoVenv
}

function Get-DefaultCMakeGenerator() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Arch
    )
    $HostArch = (Get-HostArch)
    # It's entirely possible that $Arch is "arm64ec" and $HostArch is "arm64".
    # Unfortunately, Launch-VsDevShell.ps1 doesn't support arm64ec so we cannot
    # use Ninja.
    if ($Arch -eq $HostArch) {
        "Ninja"
    } else {
        Write-Host "Cross compiling for $Arch on $HostArch host. Cannot use Ninja."
        "Visual Studio 17 2022"
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

function Get-TargetPyVersionFilePath() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )
    "$BuildDir\target-py-version-$Config.txt"
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

function Save-TargetPyVersion() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $false)]
        [string]$TargetPyVersion = ""
    )

    $TargetPyVersionFilePath = (Get-TargetPyVersionFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path "$TargetPyVersionFilePath\..")) {
        New-Item -Path "$TargetPyVersionFilePath\.." -ItemType Directory | Out-Null
    }
    $TargetPyVersion | Out-File -FilePath $TargetPyVersionFilePath
}

function Test-QairtSdkDiffers() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$QairtSdkRoot
    )

    $QairtSdkPathPath = (Get-QairtSdkFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path -Path $QairtSdkPathPath)) {
        return $True
    }

    $LastSdkPath = Get-Content -Path $QairtSdkPathPath
    return $LastSdkPath -ne $QairtSdkRoot
}

function Test-TargetPyVersionDiffers() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $false)]
        [string]$TargetPyVersion = ""
    )

    $TargetPyVersionFilePath = (Get-TargetPyVersionFilePath -BuildDir $BuildDir -Config $Config)
    if (-Not (Test-Path -Path $TargetPyVersionFilePath)) {
        return $True
    }

    $LastTargetPyVersion = Get-Content -Path $TargetPyVersionFilePath
    return $LastTargetPyVersion -ne $TargetPyVersion
}

function Test-UpdateNeeded() {
    param (
        [Parameter(Mandatory = $true)]
        [string]$BuildDir,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $false)]
        [string]$TargetPyVersion = "",
        [Parameter(Mandatory = $true)]
        [string]$QairtSdkRoot,
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

    if (Test-TargetPyVersionDiffers -BuildDir $BuildDir -Config $Config -TargetPyVersion $TargetPyVersion) {
        Write-Host "Previous build used a different Python version."
        return $True
    }

    if (Test-QairtSdkDiffers -BuildDir $BuildDir -Config $Config -QairtSdkRoot $QairtSdkRoot) {
        Write-Host "Previous build used a different QAIRT SDK."
        return $True
    }

    Write-Host "No need to update build system."
    return $False
}

function Use-PyVEnv() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PyVEnv,
        [scriptblock]$Code
    )

    if ($env:VIRTUAL_ENV) {
        $PrevVenv = $env:VIRTUAL_ENV
        Exit-PyVenv
    }

    try {
        Enter-PyVenv -PyVEnv $PyVEnv
        Invoke-Command $Code
    }
    finally {
        Exit-PyVenv
        if ($null -ne $PrevVenv) {
            Enter-PyVenv $PrevVenv
        }
    }
}

function Use-WorkingDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [scriptblock]$Code
    )

    Push-Location $Path
    try {
        Invoke-Command $Code
    }
    finally {
        Pop-Location
    }
}
