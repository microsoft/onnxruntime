# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

. "$RepoRoot\qcom\scripts\windows\utils.ps1"

function Get-ToolsDir() {
    if (Test-Path Env:ORT_BUILD_TOOLS_PATH) {
        $ToolsDir = $Env:ORT_BUILD_TOOLS_PATH
    } else {
        $ToolsDir = (Join-Path $RepoRoot "build\Tools")
    }

    # We don't use Assert-Success because this has an exit code of 1
    # when the directory already exists. Powershell somehow knows that
    # this is not a failure so we test $? directly.
    New-Item -ItemType Directory $ToolsDir -Force | Out-Null
    if (-not $?) {
        throw "Failed to create $ToolsDir"
    }
    return $ToolsDir
}

function Get-CCacheBinDir () {
    Get-PackageBinDir ccache_windows_$(Get-HostArch)
}

function Get-CMakeBinDir() {
    Get-PackageBinDir cmake_windows_$(Get-HostArch)
}

function Get-JavaBinDir() {
    Get-PackageBinDir java_windows_x86_64
}

function Get-NinjaBinDir() {
    Get-PackageBinDir ninja_windows_x86_64
}

function Get-OnnxModelsRoot() {
    Get-PackageContentDir onnx_models
}

function Get-PackageBinDir() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package whose bin directory to get.")]
        [string]$Package
    )

    Install-Package $Package
    Assert-Success -ErrorMessage "Failed to get bin directory for $Package." {
        python.exe (Get-PackageManager) --print-bin-dir --package $Package --package-root (Get-ToolsDir)
    }
}

function Get-PackageContentDir() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package whose content directory to get.")]
        [string]$Package
    )

    Install-Package $Package
    Assert-Success -ErrorMessage "Failed to get content directory for $Package." {
        python.exe (Get-PackageManager) --print-content-dir --package $Package --package-root (Get-ToolsDir)
    }
}

function Get-PackageManager() {
    Join-Path $RepoRoot "qcom\scripts\all\package_manager.py"
}

function Get-QairtRoot() {
    Get-PackageContentDir qairt
}

function Install-Package() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package to install.")]
        [string]$Package
    )

    Assert-Success -ErrorMessage "Failed to install package $Package" {
        python.exe (Get-PackageManager) --install --package $Package --package-root (Get-ToolsDir)
    }
}

# We call this "Optimize" to conform to the PowerShell approved verbs list.
function Optimize-ToolsDir() {
    Assert-Success -ErrorMessage "Failed to optimize tools directory." {
        python.exe (Get-PackageManager) --clean --package-root (Get-ToolsDir)
    }
}
