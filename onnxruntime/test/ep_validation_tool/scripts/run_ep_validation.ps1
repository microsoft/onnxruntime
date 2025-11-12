# EP Validation Script
# Updated to align with the new model_map.json structure used by setup_ep_validation.ps1
# 
# Key changes:
# - Uses model_map.json instead of ep_validation_config.json
# - Expects directory structure: ./data/<modelName>/ and ./models/<modelName>/
# - Looks for validation configs in ./validation_configs/<modelName>.json
# - Supports the new versioned dataset structure created by the setup script
# - Only uses quantized models (.onnxe) - no reference models (.onnx) are downloaded by setup

param (
    [ValidateSet('qnn', 'ov', 'vitisai')]
    [string]$executionProvider,
    [ValidateScript({ Test-Path $_ })]
    [string]$configPath = "./model_map.json",
    [ValidateScript({ Test-Path $_ })]
    [string]$exePath = "./bin/ps_onnxruntime_test.exe",
    [string]$outputDir = "./outputs",
    [ValidateScript({ Test-Path $_ })]
    [string]$validationConfigDir = "./validation_configs"
)

. ./common_utils.ps1

function Get-ModelPath {
    param (
        [string]$modelName,
        [string]$modelExtension = "onnxe"
    )

    # Search for the model file in the models/<modelName> directory
    $modelPath = "./models/$modelName"
    if (-not (Test-Path $modelPath)) {
        Write-Warning "Model directory not found: $modelPath"
        return $null
    }
    
    $modelFile = Get-ChildItem -Path $modelPath -Recurse -Filter "*.$modelExtension" | Select-Object -First 1

    if (-not $modelFile) {
        Write-Warning "No .$modelExtension file found in $modelPath"
        return $null
    }

    return $modelFile.FullName
}

function Get-DatasetPath {
    param (
        [string]$modelName
    )

    # Use the data/<modelName> directory structure created by setup script
    $dataPath = "./data/$modelName"
    if (-not (Test-Path $dataPath)) {
        Write-Warning "Dataset directory not found: $dataPath"
        return $null
    }

    return $dataPath
}

# Read and parse JSON file
$jsonContent = Get-Content -Path $configPath -Raw | ConvertFrom-Json

# Create output directory
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

# Process each model
if ($jsonContent.models) {
    $modelNames = $jsonContent.models.PSObject.Properties.Name
    
    foreach ($modelName in $modelNames) {
        Write-Host "`n========== Processing model: $modelName =========="
        
        $modelConfig = $jsonContent.models.$modelName
        
        # Get quantized model path (only quantized models are downloaded by setup script)
        $quantizedModelPath = Get-ModelPath -modelName $modelName -modelExtension "onnxe"
        
        if (-not $quantizedModelPath) {
            Write-Warning "Skipping $modelName - quantized model not found"
            continue
        }
        
        # Get dataset path
        $datasetPath = Get-DatasetPath -modelName $modelName
        if (-not $datasetPath) {
            Write-Warning "Skipping $modelName - dataset not found"
            continue
        }
        
        # Look for validation config file
        $validationConfigPath = "$validationConfigDir/$modelName.json"
        if (-not (Test-Path $validationConfigPath)) {
            Write-Warning "Validation config not found for $modelName at $validationConfigPath, skipping"
            continue
        }
        
        # Load validation config
        $validationConfig = Get-Content -Path $validationConfigPath -Raw | ConvertFrom-Json
        
        # Get encryption key from model config
        $encryptionKey = ""
        if ($modelConfig.quantized_model -and $modelConfig.quantized_model.encryption_key) {
            $encryptionKey = $modelConfig.quantized_model.encryption_key
        }
        
        # Modify session options based on whether model is encrypted (.onnxe extension)
        if ($quantizedModelPath -like "*.onnxe") {
            if ($validationConfig.arguments.sessionOptions) {
                $sessionOptionsArray = $validationConfig.arguments.sessionOptions -split ' '
                $disableCpuFallback = $sessionOptionsArray | Where-Object { $_ -like "session.disable_cpu_ep_fallback|*" }
                if ($disableCpuFallback) {
                    $validationConfig.arguments.sessionOptions = $disableCpuFallback
                } else {
                    # If not found, add it
                    $validationConfig.arguments.sessionOptions = "session.disable_cpu_ep_fallback|1"
                }
            }
        }
        
        # Convert JSON config to command-line arguments
        $configArgs = Convert-JsonToArgs -config $validationConfig
        
        # Set up output paths
        $validationOutDir = Join-Path -Path $outputDir -ChildPath $modelName
        New-Item -ItemType Directory -Path $validationOutDir -Force | Out-Null
        $validationOutPath = Join-Path -Path $validationOutDir -ChildPath "$modelName`_validation_output.json"

        # Run validation for data generation stage first
        $exeArgsDataGen = @(
            "--modelPath", "$quantizedModelPath",
            "--datasetDir", "$datasetPath",
            "--outputDir", "$validationOutDir",
            "--resultPath", "$validationOutPath",
            "--stage", "-1"
        )

        # Add encryption key if available
        if ($encryptionKey) {
            $exeArgsDataGen += "--modelKey"
            $exeArgsDataGen += "$encryptionKey"
        }

        Write-Host "`nExecuting command for data generation stage:`n$exePath $exeArgsDataGen $configArgs`n"
        & $exePath $exeArgsDataGen $configArgs

        # Then run validation for the inference stage
        $exeArgsInference = @(
            "--modelPath", "$quantizedModelPath",
            "--datasetDir", "$datasetPath",
            "--outputDir", "$validationOutDir",
            "--resultPath", "$validationOutPath",
            "--stage", "2"
        )

        if ($encryptionKey) {
            $exeArgsInference += "--modelKey"
            $exeArgsInference += "$encryptionKey"
        }

        Write-Host "`nExecuting command for inference stage:`n$exePath $exeArgsInference $configArgs`n"
        & $exePath $exeArgsInference $configArgs

        # Check if output file was created
        if (Test-Path $validationOutPath) {
            # Add model package information
            $validationOutputData = Get-Content -Raw -Path $validationOutPath | ConvertFrom-Json
            
            # Add model information from model_map.json
            if ($modelConfig.quantized_model) {
                Add-Member -InputObject $validationOutputData -MemberType NoteProperty -Name "quantized_model_package_name" -Value $modelConfig.quantized_model.package_name -Force
                Add-Member -InputObject $validationOutputData -MemberType NoteProperty -Name "quantized_model_package_version" -Value $modelConfig.quantized_model.package_version -Force
            }
            
            if ($modelConfig.dataset_package) {
                Add-Member -InputObject $validationOutputData -MemberType NoteProperty -Name "dataset_package_name" -Value $modelConfig.dataset_package.package_name -Force
                Add-Member -InputObject $validationOutputData -MemberType NoteProperty -Name "dataset_package_version" -Value $modelConfig.dataset_package.package_version -Force
            }
            
            Add-Member -InputObject $validationOutputData -MemberType NoteProperty -Name "model_name" -Value $modelName -Force
            
            ConvertTo-Json @($validationOutputData) -Depth 10 | Format-Json | Set-Content $validationOutPath -Encoding UTF8

            # Get device configuration
            $deviceConfig = Get-DeviceConfig $exePath $executionProvider

            # Print device configuration to console
            Print-DeviceConfig $deviceConfig

            # Dump device configuration to test report json
            Dump-DeviceConfig $deviceConfig -outputPath $validationOutPath
        } else {
            Write-Warning "Validation output file was not created for $modelName"
        }
    }
} else {
    Write-Warning "No models found in configuration file"
}

Write-Host "`n========== Validation completed ==========="