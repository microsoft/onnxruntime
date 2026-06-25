# select_dml_package.ps1
# Helper script to select the correct DML NuGet package based on build type
# Usage: select_dml_package.ps1 -SourceDir <path> -IsReleaseBuild <true|false> -Action <copy|rename> [-DestinationDir <path>] [-NewName <name>]

param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir,

    [Parameter(Mandatory=$true)]
    [string]$IsReleaseBuild,

    [Parameter(Mandatory=$true)]
    [ValidateSet("copy", "rename")]
    [string]$Action,

    [Parameter(Mandatory=$false)]
    [string]$DestinationDir,

    [Parameter(Mandatory=$false)]
    [string]$NewName
)

$ErrorActionPreference = "Stop"

Write-Host "Searching for packages in: $SourceDir"
Write-Host "IsReleaseBuild: $IsReleaseBuild"
Write-Host "Action: $Action"

# Convert string to boolean
$isRelease = [System.Convert]::ToBoolean($IsReleaseBuild)

# Find all matching packages
$allPackages = Get-ChildItem -Path $SourceDir -Filter "Microsoft.ML.OnnxRuntime.DirectML.*.nupkg"
Write-Host "Found $($allPackages.Count) total package(s):"
$allPackages | ForEach-Object { Write-Host "  - $($_.Name)" }

# Filter packages based on build type
$filteredPackages = $allPackages | Where-Object {
    $name = $_.Name
    $isSymbols = $name -like "*symbols*"
    $isDev = $name -like "*-dev*"

    if ($isSymbols) {
        return $false
    }

    if ($isRelease) {
        return -not $isDev
    } else {
        return $isDev
    }
}

Write-Host "After filtering (isRelease=$isRelease), found $($filteredPackages.Count) matching package(s):"
$filteredPackages | ForEach-Object { Write-Host "  - $($_.Name)" }

if ($filteredPackages.Count -eq 0) {
    Write-Error "No matching package found!"
    exit 1
}

# Select the first matching package (sorted by name length for consistency)
$selectedPackage = $filteredPackages | Sort-Object { $_.Name.Length } | Select-Object -First 1
Write-Host "Selected package: $($selectedPackage.FullName)"

# Perform the action
if ($Action -eq "copy") {
    if (-not $DestinationDir) {
        Write-Error "DestinationDir is required for copy action"
        exit 1
    }
    Write-Host "Copying to: $DestinationDir"
    Copy-Item -Path $selectedPackage.FullName -Destination $DestinationDir -Force
    Write-Host "Copy successful."
}
elseif ($Action -eq "rename") {
    if (-not $NewName) {
        Write-Error "NewName is required for rename action"
        exit 1
    }
    Write-Host "Renaming to: $NewName"
    Rename-Item -Path $selectedPackage.FullName -NewName $NewName -Force
    Write-Host "Rename successful."
}

exit 0
