[CmdletBinding()]
param (
    # The major version of Node.js to use. Example: '20'
    [Parameter(Mandatory = $true)]
    [string]$MajorVersion
)

try {
    # Get the processor architecture ID using CIM
    # 9 = x64, 12 = arm64
    $architectureId = (Get-CimInstance -ClassName Win32_Processor).Architecture

    # Map the architecture ID to the string used in the tool path
    $archString = switch ($architectureId) {
        9       { "x64" }
        12      { "arm64" }
        default { throw "Unsupported CPU architecture: $architectureId. This script only supports x64 and arm64." }
    }

    Write-Host "Detected Architecture: $archString"

    # --- New Logic to find the latest version ---
    $nodeVersionsPath = Join-Path $env:AGENT_TOOLSDIRECTORY "node"
    if (-not (Test-Path -Path $nodeVersionsPath)) {
        throw "Node.js tool directory not found at '$nodeVersionsPath'."
    }

    # Find all directory names matching the major version (e.g., "20.*")
    $matchingVersions = Get-ChildItem -Path $nodeVersionsPath |
                        Where-Object { $_.PSIsContainer -and $_.Name -like "$MajorVersion.*" } |
                        Select-Object -ExpandProperty Name

    if ($null -eq $matchingVersions) {
        throw "No installed Node.js versions found for major version '$MajorVersion' at '$nodeVersionsPath'."
    }

    # Sort the versions to find the highest one and select it
    $latestVersion = $matchingVersions | Sort-Object -Descending {[version]$_} | Select-Object -First 1
    Write-Host "Found latest matching version: $latestVersion"
    # --- End of New Logic ---

    # Construct the full path using the discovered latest version
    $nodeToolPath = Join-Path $nodeVersionsPath "$latestVersion\$archString"

    # Verify that the final directory exists
    if (-not (Test-Path -Path $nodeToolPath -PathType Container)) {
        throw "Node.js tool path not found. Please ensure version '$latestVersion' for '$archString' exists at: $nodeToolPath"
    }

    # Use the Azure DevOps logging command to prepend the directory to the PATH
    Write-Host "##vso[task.prependpath]$nodeToolPath"
    Write-Host "Successfully added Node.js $latestVersion ($archString) to the PATH."

}
catch {
    # If any error occurs, log it as an error in the pipeline and fail the task
    Write-Host "##vso[task.logissue type=error]$($_.Exception.Message)"
    exit 1
}