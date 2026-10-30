# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

param(
    [Parameter(Mandatory=$true)]
    [string]$PackScript
)

$tokens=$null
$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($PackScript, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) {
    throw "Failed to parse ${PackScript}: $($errors[0].Message)"
}

$functionAst=$ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Write-Node-NuGet-Integrity-Metadata"
}, $true)
if (-not $functionAst) {
    throw "Write-Node-NuGet-Integrity-Metadata was not found in ${PackScript}"
}
Invoke-Expression $functionAst.Extent.Text

Function Invoke-RestMethod {
    throw "fixture feed unavailable"
}

foreach ($case in @(
    @{ Mode="release"; Version="1.32.0"; Feed="nuget" },
    @{ Mode="dev"; Version="1.32.0-dev-20261029-abc"; Feed="nuget_nightly" }
)) {
    $temp=Join-Path ([System.IO.Path]::GetTempPath()) "ort-node-integrity-$([guid]::NewGuid())"
    try {
        $scriptDir=Join-Path $temp "node/script"
        $signedDir=Join-Path $temp "signed"
        New-Item -ItemType Directory -Path $scriptDir,$signedDir -Force | Out-Null

        $packageName="Microsoft.ML.OnnxRuntime.Gpu.Linux.$($case.Version).nupkg"
        [System.IO.File]::WriteAllText(
            (Join-Path $signedDir $packageName),
            "signed package fixture")
        [System.IO.File]::WriteAllText(
            (Join-Path $scriptDir "install-metadata-versions.js"),
            "module.exports={nuget:[{feed:'$($case.Feed)',version:'$($case.Version)'},{feed:'nuget',version:'0.0.0-unavailable'}]};")

        $TARGET="node"
        $MODE=$case.Mode
        $JS_TARGET_DIR=Join-Path $temp "node"
        $SIGNED_NUGET_DIR=$signedDir
        Write-Node-NuGet-Integrity-Metadata

        $versions=Get-Content -Raw (Join-Path $scriptDir "install-metadata-versions.js")
        $integrity=Get-Content -Raw (Join-Path $scriptDir "install-metadata-integrity.js")
        if ($versions -notmatch [regex]::Escape($case.Feed) -or
            $versions -notmatch [regex]::Escape($case.Version) -or
            $integrity -notmatch "sha512-") {
            throw "Integrity metadata fixture failed for $($case.Mode)"
        }
    } finally {
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Signed release and nightly integrity metadata fixtures passed."
