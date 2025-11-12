param (
    [ValidateScript({ Test-Path $_ })]
    [string]$modelDir,
    [ValidateScript({ Test-Path $_ })]
    [string]$datasetDir,
    [ValidateScript({ Test-Path $_ })]
    [string]$configPath,
    [Parameter(Mandatory=$true)]
    [ValidateSet('qnn', 'ov', 'vitisai')]
    [string]$executionProvider,
    [Parameter(Mandatory=$true)]
    [ValidateScript({ Test-Path $_ })]
    [string]$exePath,
    [ValidateSet('-1', '0', '1', '2')]
    [string]$stage = "2",
    [string]$modelKey = "",
    [string]$outputDir = "./outputs"
)

. ./common_utils.ps1

# ----------------------------------------
# Main Script Execution
# ----------------------------------------

# Load test config json
$config = Load-JsonConfig -configPath $configPath

# Modify session options based on whether model is encrypted
if ($modelKey -ne "") {
    # For encrypted models, only use session.disable_cpu_ep_fallback|1
    if ($config.arguments.sessionOptions) {
        $sessionOptionsArray = $config.arguments.sessionOptions -split ' '
        $disableCpuFallback = $sessionOptionsArray | Where-Object { $_ -like "session.disable_cpu_ep_fallback|*" }
        if ($disableCpuFallback) {
            $config.arguments.sessionOptions = $disableCpuFallback
        } else {
            # If not found, add it
            $config.arguments.sessionOptions = "session.disable_cpu_ep_fallback|1"
        }
    }
}

# Search for model files
if ($modelKey -eq "") {
    $qdqModel = Get-ChildItem -Path $modelDir -Filter "*.quant.onnx" -Recurse | Select-Object -First 1
} else {
    $qdqModel = Get-ChildItem -Path $modelDir -Filter "*.onnxe" -Recurse | Select-Object -First 1
}

# Find reference model - exclude .quant.onnx files (optional)
$refModel = Get-ChildItem -Path $modelDir -Filter "*.onnx" -Recurse | Where-Object { $_.Name -notlike "*.quant.onnx" } | Select-Object -First 1

# Handle the case where no model is found
if (-not $qdqModel) {
    $modelType = if ($modelKey -eq "") { ".quant.onnx" } else { ".onnxe" }
    Write-Host "No $modelType model found in $modelDir"
    exit 1
}

# Log reference model status
if ($refModel) {
    Write-Host "Reference model found: $($refModel.Name)"
} else {
    Write-Host "No reference model found - proceeding without reference model"
}

# Convert JSON config to command-line arguments
$configArgs = Convert-JsonToArgs -config $config

# Create output directory if it doesn't exist
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

# Generate path to output file
$outputPath = Join-Path -Path $outputDir -ChildPath "test_result.json"

# Add model and output folder paths
$exeArgs = @()
$exeArgs += "--modelPath"
$exeArgs += $qdqModel.FullName

# Only add reference model if it exists
if ($refModel) {
    $exeArgs += "--refModelPath"
    $exeArgs += $refModel.FullName
}

$exeArgs += "--datasetDir"
$exeArgs += "$datasetDir"
$exeArgs += "--outputDir"
$exeArgs += $outputDir
$exeArgs += "--resultPath"
$exeArgs += "$outputPath"
$exeArgs += "--stage"
$exeArgs += "$stage"

if ($modelKey)
{
    $exeArgs += "--modelKey"
    $exeArgs += "$modelKey"
}

# Print the command
Write-Host "`nExecuting command:`n$exePath $exeArgs $configArgs`n"

# Call ps_onnxruntime_test.exe with config arguments
& $exePath $exeArgs $configArgs

# Get device configuration
$deviceConfig = Get-DeviceConfig $exePath $executionProvider

# Print device configuration to console
Print-DeviceConfig $deviceConfig

# Dump device configuration to test report json
Dump-DeviceConfig $deviceConfig -outputPath $outputPath
