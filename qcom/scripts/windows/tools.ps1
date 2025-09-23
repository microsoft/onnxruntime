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

function Get-PythonBinDir() {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Version,
        [Parameter(Mandatory = $true)]
        [string]$Arch
    )

    switch ($Arch) {
        "arm64" { $PyVsn = "$Version-arm64"; $PkgArch = "arm64" }
        "arm64ec" { $PyVsn = "$Version"; $PkgArch = "x86_64" }
        "x86_64" { $PyVsn = $Version; $PkgArch = "x86_64" }
        Default { throw "Unknown Python executable arch $Arch "}
    }

    if ((Get-HostArch) -eq "x86_64" -and $Arch -eq "arm64") {
        throw "Python for $Arch not supported on x86_64 host."
    }

    # An appropriate Python might exist on the system already. If so, let's use it
    # so we don't clobber something someone might be using.
    $PythonExePath = (py "-$PyVsn" -c "import sys; print(sys.executable)")

    $PkgName = "python_$($Version.Replace('.', ''))_windows_$PkgArch"

    # See https://docs.python.org/3/using/windows.html#return-codes
    # 101 --> Failed to launch Python
    # 103 --> Unable to locate the requested version
    switch ($LASTEXITCODE) {
        0 {
            Write-Host "Using existing $PyVsn in $PythonExePath"
            return (Resolve-Path (Split-Path -Parent $PythonExePath))
        }
        103 {
            Write-Host "Installing $PkgName"
            return Get-PackageBinDir $PkgName
        }
        101 {
            Write-Host "Repairing $PkgName"
            Repair-Package $PkgName
            return Get-PackageBinDir $PkgName
        }
        Default {
            throw "Could not locate/install Python $Version for $Arch."
        }
    }
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

function Repair-Package() {
    param(
        [Parameter(Mandatory = $true,
        HelpMessage = "The package to repair.")]
        [string]$Package
    )

    Assert-Success -ErrorMessage "Failed to repair package $Package" {
        python.exe (Get-PackageManager) --repair --package $Package --package-root (Get-ToolsDir)
    }
}
