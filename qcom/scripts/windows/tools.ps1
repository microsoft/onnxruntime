# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

function Get-ToolsDir() {
    $ToolsDir = (Join-Path $RepoRoot "build\Tools")
    New-Item -ItemType Directory $ToolsDir -Force
}

function Install-Package() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package to install.")]
        [string]$Package
    )

    $PackageManager = (Join-Path $RepoRoot "qcom\scripts\all\package_manager.py")
    python.exe $PackageManager --install --package $Package --package-root (Get-ToolsDir)
    python.exe $PackageManager --print-content-dir --package $Package --package-root (Get-ToolsDir)
}
