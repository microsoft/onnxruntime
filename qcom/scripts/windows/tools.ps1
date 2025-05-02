# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RepoRoot = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)\..\..\..").Path

function Get-NugetPath() {
    Join-Path (Get-ToolsDir) "nuget.exe"
}

function Get-ToolsDir() {
    $ToolsDir = (Join-Path $RepoRoot "build\Tools")
    New-Item -ItemType Directory $ToolsDir -Force
}

function Install-Nuget() {
    $NugetPath = (Get-NugetPath)
    if (-Not (Test-Path -Path $NugetPath)) {
        $NuGetUrl = "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
        Invoke-WebRequest -Uri $NuGetUrl -OutFile $NugetPath

        if (-Not $?) {
            throw "Failed to fetch NuGet."
        }
    }

    $NugetPath
}
