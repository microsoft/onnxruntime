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

Function Get-Fixture-Sha512Integrity([string]$Path) {
    $hash=(Get-FileHash -Path $Path -Algorithm SHA512).Hash
    $bytes=[byte[]]::new($hash.Length / 2)
    for ($i=0; $i -lt $bytes.Length; $i++) {
        $bytes[$i]=[Convert]::ToByte($hash.Substring($i * 2, 2), 16)
    }
    return "sha512-$([Convert]::ToBase64String($bytes))"
}

Function Invoke-RestMethod {
    param(
        [string]$Uri,
        [switch]$UseBasicParsing
    )
    $null=$Uri
    $null=$UseBasicParsing
    return @{
        resources=@(
            @{
                '@type'='PackageBaseAddress/3.0.0'
                '@id'='fixture://flat-container/'
            }
        )
    }
}

Function Invoke-WebRequest {
    param(
        [string]$Uri,
        [string]$OutFile,
        [switch]$UseBasicParsing
    )
    $null=$UseBasicParsing

    if ($Uri -notmatch [regex]::Escape($script:FixtureVersion) -or
        $Uri -match '0\.0\.0-unavailable') {
        throw "fixture feed unavailable"
    }
    Copy-Item -LiteralPath $script:FixtureFeedPackage -Destination $OutFile
}

foreach ($case in @(
    @{ Mode="release"; Version="1.32.0"; Feed="nuget"; ExpectedSource="feed" },
    @{ Mode="dev"; Version="1.32.0-dev-20261029-abc"; Feed="nuget_nightly"; ExpectedSource="signed" }
)) {
    $temp=Join-Path ([System.IO.Path]::GetTempPath()) "ort-node-integrity-$([guid]::NewGuid())"
    try {
        $scriptDir=Join-Path $temp "node/script"
        $signedDir=Join-Path $temp "signed"
        New-Item -ItemType Directory -Path $scriptDir,$signedDir -Force | Out-Null

        $packageName="Microsoft.ML.OnnxRuntime.Gpu.Linux.$($case.Version).nupkg"
        $signedPackage=Join-Path $signedDir $packageName
        $script:FixtureFeedPackage=Join-Path $temp "feed-$packageName"
        $script:FixtureVersion=$case.Version
        [System.IO.File]::WriteAllText(
            $signedPackage,
            "signed package fixture")
        [System.IO.File]::WriteAllText(
            $script:FixtureFeedPackage,
            "feed-served package fixture")
        [System.IO.File]::WriteAllText(
            (Join-Path $scriptDir "install-metadata-versions.js"),
            "module.exports={nuget:[{feed:'$($case.Feed)',version:'$($case.Version)'},{feed:'nuget',version:'0.0.0-unavailable'}]};")

        $TARGET="node"
        $MODE=$case.Mode
        $JS_TARGET_DIR=Join-Path $temp "node"
        $SIGNED_NUGET_DIR=$signedDir
        Write-Node-NuGet-Integrity-Metadata

        $versionsPath=(Join-Path $scriptDir "install-metadata-versions.js").Replace('\', '/')
        $integrityPath=(Join-Path $scriptDir "install-metadata-integrity.js").Replace('\', '/')
        $versions=node -p "JSON.stringify(require('$versionsPath'))" | ConvertFrom-Json
        $integrity=node -p "JSON.stringify(require('$integrityPath'))" | ConvertFrom-Json
        $candidates=@($versions.nuget)
        if ($candidates.Count -ne 1 -or
            $candidates[0].feed -ne $case.Feed -or
            $candidates[0].version -ne $case.Version) {
            throw "Unexpected version candidates for $($case.Mode): $($candidates | ConvertTo-Json -Compress)"
        }

        $expectedKey="microsoft.ml.onnxruntime.gpu.linux@$($case.Version.ToLower())"
        $integrityEntries=@($integrity.PSObject.Properties)
        $expectedPackage=if ($case.ExpectedSource -eq "feed") {
            $script:FixtureFeedPackage
        } else {
            $signedPackage
        }
        $expectedIntegrity=Get-Fixture-Sha512Integrity $expectedPackage
        if ($integrityEntries.Count -ne 1 -or
            $integrityEntries[0].Name -ne $expectedKey -or
            $integrityEntries[0].Value -ne $expectedIntegrity) {
            throw "Unexpected integrity metadata for $($case.Mode): $($integrity | ConvertTo-Json -Compress)"
        }
    } finally {
        $script:FixtureFeedPackage=$null
        $script:FixtureVersion=$null
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Feed-served release and signed nightly integrity metadata fixtures passed."
