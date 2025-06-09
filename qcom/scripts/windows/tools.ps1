# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

function Get-ToolsDir() {
    $ToolsDir = (Join-Path $RepoRoot "build\Tools")
    New-Item -ItemType Directory $ToolsDir -Force
}

function Get-AndroidNdkRoot() {
    $InstallNdk = "$RepoRoot\qcom\scripts\all\install_ndk.py"

    python.exe $InstallNdk --cli-tools-root (Get-PackageContentDir android_commandlinetools_windows_x86_64)
    if (-not $?) {
        throw "Failed to get NDK content root."
    }
}

function Get-AndroidSdkRoot() {
    (Resolve-Path "$(Get-PackageContentDir android_commandlinetools_windows_x86_64)\..").Path
}

function Get-PackageBinDir() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package whose bin directory to get.")]
        [string]$Package
    )

    Install-Package $Package
    python.exe (Get-PackageManager) --print-bin-dir --package $Package --package-root (Get-ToolsDir)
    if (-not $?) {
        throw "Failed to get bin directory for $Package."
    }
}

function Get-PackageContentDir() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package whose content directory to get.")]
        [string]$Package
    )

    Install-Package $Package
    python.exe (Get-PackageManager) --print-content-dir --package $Package --package-root (Get-ToolsDir)
    if (-not $?) {
        throw "Failed to get content directory for $Package."
    }
}

function Get-PackageManager() {
    Join-Path $RepoRoot "qcom\scripts\all\package_manager.py"
}

function Install-Package() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package to install.")]
        [string]$Package
    )

    python.exe (Get-PackageManager) --install --package $Package --package-root (Get-ToolsDir)
    if (-not $?) {
        throw "Failed to install package $Package."
    }
}
