param (
    [ValidateScript({ Test-Path $_ })]
    [string]$configPath = "./model_map.json"
)

. ./common_utils.ps1

function Move-Models {
    param (
        [string]$outputPath,
        [string]$modelName
    )

    # Move *.onnxe and *.onnx files to the models/<modelName> directory
    $targetPath = "./models/$modelName"
    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
    
    Get-ChildItem -Path $outputPath -Recurse -Include *.onnxe, *.onnx | ForEach-Object {
        # Extract the file extension (.onnx or .onnxe)
        $extension = if ($_.Extension -eq ".onnxe") { ".onnxe" } else { ".onnx" }
        
        $newFileName = "$modelName.quant$extension"
        
        $destinationPath = Join-Path $targetPath $newFileName
        Move-Item -Path $_.FullName -Destination $destinationPath -Force
        Write-Host "  Renamed and moved: $($_.Name) -> $newFileName"
    }

    Get-ChildItem -Path $outputPath -Recurse -Include *onnx.data | ForEach-Object {        
        $destinationPath = Join-Path $targetPath $_.Name
        Move-Item -Path $_.FullName -Destination $destinationPath -Force
        Write-Host "  Moved: $($_.Name) "
    }

    # Remove the original output directory
    Remove-Item -Path $outputPath -Recurse -Force
}

function Download-UPack {
    param (
        [string]$feed,
        [string]$name,
        [string]$version,
        [string]$outputPath
    )

    # Define the command and arguments
    $command = "az"
    $arguments = @(
        "artifacts", "universal", "download",
        "--organization", "https://devicesasg.visualstudio.com/",
        "--project", "eb75968e-53cb-42c1-80cd-736987b5e259",
        "--scope", "project",
        "--feed", "$feed",
        "--name", "$name",
        "--version", "$version",
        "--path", "$outputPath"
    ) 
    Write-Host "`nDownloading UPack dataset package:`n  Feed: $feed`n  Name: $name`n  Version: $version`n"
    & $command $arguments
}

function Install-PythonRequirements {
    param (
        [string]$requirementsPath = "./requirements.txt"
    )
    
    if (-not (Test-Path $requirementsPath)) {
        Write-Warning "Requirements file not found: $requirementsPath"
        return
    }
    
    Write-Host "Installing Python packages from requirements.txt..."
    
    # Install packages using pip
    & python -m pip install -r $requirementsPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully installed Python packages from requirements.txt"
    } else {
        Write-Warning "Failed to install some packages from requirements.txt. Some operations may not work correctly."
    }
}

function Convert-RawToNpy {
    param (
        [string]$folderPath,
        [string]$shape,
        [string]$dtype = "float32"
    )
    
    # Check if folder contains any raw files (non-npy files)
    $rawFiles = Get-ChildItem -Path $folderPath -File | Where-Object { $_.Extension -ne ".npy" }
    
    if ($rawFiles.Count -eq 0) {
        Write-Host "  No raw files found in $folderPath, skipping conversion."
        return
    }
    
    Write-Host "  Converting $($rawFiles.Count) raw files to npy format in: $folderPath"
    Write-Host "    Data type: $dtype, Shape: $shape"
    
    # Call the Python script with folder mode
    $scriptPath = Join-Path $PSScriptRoot "raw_to_npy.py"
    if (-not (Test-Path $scriptPath)) {
        Write-Warning "  raw_to_npy.py script not found at: $scriptPath"
        return
    }
    
    $pythonArgs = @(
        $scriptPath,
        $folderPath,
        $dtype,
        $shape,
        "--folder_mode"
    )
    
    # Execute Python script
    $result = & python $pythonArgs 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Successfully converted raw files to npy format"
    } else {
        Write-Warning "  Failed to convert some files. Python output:"
        Write-Warning ($result -join "`n")
    }
}

function Process-DatasetConversion {
    param (
        [string]$dataPath,
        [string]$modelName,
        [string]$version
    )
    
    # Look for Model_<modelName>.json file in the data folder
    $modelConfigPath = "$dataPath/Model_$($modelName.ToUpper()).json"
    if (-not (Test-Path $modelConfigPath)) {
        Write-Host "Model config file not found: $modelConfigPath, skipping raw-to-npy conversion"
        return
    }
    
    Write-Host "Processing raw-to-npy conversion using config: $modelConfigPath"
    
    # Read model config
    $modelConfig = Get-Content -Path $modelConfigPath -Raw | ConvertFrom-Json
    
    # Read dataset config to get folder mappings
    $datasetConfigPath = "$dataPath/dataset_config.json"
    if (-not (Test-Path $datasetConfigPath)) {
        Write-Warning "Dataset config not found, skipping conversion"
        return
    }
    
    $datasetConfig = Get-Content -Path $datasetConfigPath -Raw | ConvertFrom-Json
    
    # Process input folders
    if ($modelConfig.inputs -and $modelConfig.input_dims -and $datasetConfig.input_to_dir) {
        $inputFolders = $datasetConfig.input_to_dir.PSObject.Properties
        
        foreach ($folder in $inputFolders) {
            $folderName = $folder.Value
            $tensorName = $folder.Name
            
            # Find the index of this input in the model config
            $inputIndex = -1
            for ($i = 0; $i -lt $modelConfig.inputs.Count; $i++) {
                if ($modelConfig.inputs[$i] -eq $tensorName) {
                    $inputIndex = $i
                    break
                }
            }
            
            if ($inputIndex -ge 0 -and $inputIndex -lt $modelConfig.input_dims.Count) {
                $shape = ($modelConfig.input_dims[$inputIndex] -join ",")
                $folderPath = "$dataPath/input_data/$version/$folderName"
                
                if (Test-Path $folderPath) {
                    Write-Host "`nConverting input data in folder: $folderName"
                    Convert-RawToNpy -folderPath $folderPath -shape $shape
                }
            }
        }
    }
    
    # Process output folders
    if ($modelConfig.outputs -and $modelConfig.output_dims -and $datasetConfig.output_to_dir) {
        $outputFolders = $datasetConfig.output_to_dir.PSObject.Properties
        
        foreach ($folder in $outputFolders) {
            $folderName = $folder.Value
            $tensorName = $folder.Name
            
            # Find the index of this output in the model config
            $outputIndex = -1
            for ($i = 0; $i -lt $modelConfig.outputs.Count; $i++) {
                if ($modelConfig.outputs[$i] -eq $tensorName) {
                    $outputIndex = $i
                    break
                }
            }
            
            if ($outputIndex -ge 0 -and $outputIndex -lt $modelConfig.output_dims.Count) {
                $shape = ($modelConfig.output_dims[$outputIndex] -join ",")
                $folderPath = "$dataPath/output_data/$version/$folderName"
                
                if (Test-Path $folderPath) {
                    Write-Host "`nConverting output data in folder: $folderName"
                    Convert-RawToNpy -folderPath $folderPath -shape $shape
                }
            }
        }
    }
}

function Organize-DatasetFiles {
    param (
        [string]$dataPath,
        [string]$modelName,
        [string]$version
    )

    # Check if dataset config exists and organize files accordingly
    $datasetConfigPath = "$dataPath/dataset_config.json"
    
    if (Test-Path -Path $datasetConfigPath) {
        $datasetConfig = Get-Content -Path $datasetConfigPath -Raw | ConvertFrom-Json
        
        # Update version fields in dataset config
        if ($datasetConfig.PSObject.Properties.Name -contains "input_data_version") {
            $datasetConfig.input_data_version = $version
        }
        if ($datasetConfig.PSObject.Properties.Name -contains "output_data_version") {
            $datasetConfig.output_data_version = $version
        }
        
        # Save the updated dataset config
        $datasetConfig | ConvertTo-Json -Depth 10 | Set-Content -Path $datasetConfigPath -Force
        Write-Host "Updated dataset config versions to: $version"
        
        # Organize input data files
        if ($datasetConfig.input_to_dir) {
            $inputVersionPath = "$dataPath/input_data/$version"
            
            # Check if input_data version folder contains only files (not folders)
            $inputDataItems = Get-ChildItem -Path $inputVersionPath -ErrorAction SilentlyContinue
            if ($inputDataItems) {
                $onlyFiles = ($inputDataItems | Where-Object { -not $_.PSIsContainer }).Count -eq $inputDataItems.Count
                
                if ($onlyFiles -and $inputDataItems.Count -gt 0) {
                    Write-Host "Organizing input data files into folders based on dataset config..."
                    
                    # Get folder names from input_to_dir values - ensure array handling
                    $folderNames = @()
                    $datasetConfig.input_to_dir.PSObject.Properties | ForEach-Object {
                        $folderNames += $_.Value
                    }
                    $folderNames = @($folderNames | Select-Object -Unique)
                    
                    if ($folderNames.Count -eq 1) {
                        # Single folder - use the string directly without indexing
                        $folderName = if ($folderNames -is [array]) { $folderNames[0] } else { $folderNames }
                        $targetFolder = "$inputVersionPath/$folderName"
                        New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
                        
                        Get-ChildItem -Path $inputVersionPath -File | ForEach-Object {
                            Move-Item -Path $_.FullName -Destination $targetFolder -Force
                        }
                        Write-Host "  Moved all input files to folder: $folderName"
                    }
                    elseif ($folderNames.Count -gt 1) {
                        # Multiple folders - distribute files by prefix
                        foreach ($folderName in $folderNames) {
                            $targetFolder = "$inputVersionPath/$folderName"
                            New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
                        }
                        
                        # Move files based on prefix matching
                        Get-ChildItem -Path $inputVersionPath -File | ForEach-Object {
                            $fileName = $_.Name
                            $moved = $false
                            
                            foreach ($folderName in $folderNames) {
                                if ($fileName -like "$folderName*") {
                                    Move-Item -Path $_.FullName -Destination "$inputVersionPath/$folderName" -Force
                                    $moved = $true
                                    break
                                }
                            }
                            
                            if (-not $moved) {
                                Write-Warning "  Could not determine folder for file: $fileName"
                            }
                        }
                        Write-Host "  Moved all input files to folders: $folderNames"
                    }
                } else {
                    Write-Host "Input data already organized in folders or no files found."
                }
            }
        }
        
        # Organize output data files
        if ($datasetConfig.output_to_dir) {
            $outputVersionPath = "$dataPath/output_data/$version"
            
            # Check if output_data version folder contains only files (not folders)
            $outputDataItems = Get-ChildItem -Path $outputVersionPath -ErrorAction SilentlyContinue
            if ($outputDataItems) {
                $onlyFiles = ($outputDataItems | Where-Object { -not $_.PSIsContainer }).Count -eq $outputDataItems.Count
                
                if ($onlyFiles -and $outputDataItems.Count -gt 0) {
                    Write-Host "Organizing output data files into folders based on dataset config..."
                    
                    # Get folder names from output_to_dir values - ensure array handling
                    $folderNames = @()
                    $datasetConfig.output_to_dir.PSObject.Properties | ForEach-Object {
                        $folderNames += $_.Value
                    }
                    $folderNames = @($folderNames | Select-Object -Unique)
                    
                    if ($folderNames.Count -eq 1) {
                        # Single folder - use the string directly without indexing
                        $folderName = if ($folderNames -is [array]) { $folderNames[0] } else { $folderNames }
                        $targetFolder = "$outputVersionPath/$folderName"
                        New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
                        
                        Get-ChildItem -Path $outputVersionPath -File | ForEach-Object {
                            Move-Item -Path $_.FullName -Destination $targetFolder -Force
                        }
                        Write-Host "  Moved all output files to folder: $folderName"
                    }
                    elseif ($folderNames.Count -gt 1) {
                        # Multiple folders - distribute files by prefix
                        foreach ($folderName in $folderNames) {
                            $targetFolder = "$outputVersionPath/$folderName"
                            New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
                        }
                        
                        # Move files based on prefix matching
                        Get-ChildItem -Path $outputVersionPath -File | ForEach-Object {
                            $fileName = $_.Name
                            $moved = $false
                            
                            foreach ($folderName in $folderNames) {
                                if ($fileName -like "$folderName*") {
                                    Move-Item -Path $_.FullName -Destination "$outputVersionPath/$folderName" -Force
                                    $moved = $true
                                    break
                                }
                            }
                            
                            if (-not $moved) {
                                Write-Warning "  Could not determine folder for file: $fileName"
                            }
                        }
                        Write-Host "  Moved all output files to folders: $folderNames"
                    }
                } else {
                    Write-Host "Output data already organized in folders or no files found."
                }
            }
        }
    }
    
    # Convert raw files to npy after organizing
    Process-DatasetConversion -dataPath $dataPath -modelName $modelName -version $version
}

function Process-CombinedPackage {
    param (
        [string]$tempPath,
        [string]$dataPath,
        [string]$modelName,
        [string]$version
    )
    
    Write-Host "`nProcessing combined package (dataset + model)..."
    
    # Create data directory structure
    New-Item -ItemType Directory -Path "$dataPath" -Force | Out-Null
    New-Item -ItemType Directory -Path "$dataPath/input_data/$version" -Force | Out-Null
    New-Item -ItemType Directory -Path "$dataPath/output_data/$version" -Force | Out-Null
    
    # Copy dataset config
    $datasetConfigSource = "./dataset_configs/$modelName.json"
    if (Test-Path -Path $datasetConfigSource) {
        Copy-Item -Path $datasetConfigSource -Destination "$dataPath/dataset_config.json" -Force
        Write-Host "Copied dataset config for $modelName"
    } else {
        Write-Warning "Dataset config not found: $datasetConfigSource"
    }
    
    # Copy Model_<modelName>.json if it exists
    $modelConfigSource = Get-ChildItem -Path $tempPath -Recurse -Filter "Model_$($modelName.ToUpper()).json" | Select-Object -First 1
    if ($modelConfigSource) {
        Copy-Item -Path $modelConfigSource.FullName -Destination "$dataPath/Model_$($modelName.ToUpper()).json" -Force
        Write-Host "Copied model config: Model_$($modelName.ToUpper()).json"
    }
    
    # Process input_data folder
    $inputDataSource = Get-ChildItem -Path $tempPath -Recurse -Directory -Filter "input_data" | Select-Object -First 1
    if ($inputDataSource) {
        Write-Host "Processing input_data..."
        # Copy all input_data contents directly
        Copy-Item -Path "$($inputDataSource.FullName)/*" -Destination "$dataPath/input_data/$version/" -Recurse -Force
    }
    
    # Process output_data folder
    $outputDataSource = Get-ChildItem -Path $tempPath -Recurse -Directory -Filter "output_data" | Select-Object -First 1
    if ($outputDataSource) {
        Write-Host "Processing output_data..."
        
        # Check for fp32_cpu and qdq_cpu subfolders in output_data
        $fp32Folder = Get-ChildItem -Path $outputDataSource.FullName -Directory -Filter "fp32_cpu" -ErrorAction SilentlyContinue
        $qdqFolder = Get-ChildItem -Path $outputDataSource.FullName -Directory -Filter "qdq_cpu" -ErrorAction SilentlyContinue
        
        if ($fp32Folder) {
            Write-Host "  Found fp32_cpu folder, copying contents to output_data/$version/"
            Copy-Item -Path "$($fp32Folder.FullName)/*" -Destination "$dataPath/output_data/$version/" -Recurse -Force
        } else {
            # No fp32_cpu folder, copy all output_data contents
            Copy-Item -Path "$($outputDataSource.FullName)/*" -Destination "$dataPath/output_data/$version/" -Recurse -Force
        }
    }
    
    # Process l2_norm folder if it exists
    $l2NormSource = Get-ChildItem -Path $tempPath -Recurse -Directory -Filter "l2_norm" | Select-Object -First 1
    if ($l2NormSource) {
        Write-Host "Processing l2_norm..."
        New-Item -ItemType Directory -Path "$dataPath/l2_norm" -Force | Out-Null
        Copy-Item -Path "$($l2NormSource.FullName)/*" -Destination "$dataPath/l2_norm/" -Recurse -Force
    }
    
    # Update dataset_config.json with correct version
    $datasetConfigPath = "$dataPath/dataset_config.json"
    if (Test-Path $datasetConfigPath) {
        $datasetConfig = Get-Content -Path $datasetConfigPath -Raw | ConvertFrom-Json
        
        # Update version
        if ($datasetConfig.PSObject.Properties.Name -contains "input_data_version") {
            $datasetConfig.input_data_version = $version
        }
        if ($datasetConfig.PSObject.Properties.Name -contains "output_data_version") {
            $datasetConfig.output_data_version = $version
        }
        
        # Save updated config
        $datasetConfig | ConvertTo-Json -Depth 10 | Set-Content -Path $datasetConfigPath -Force
        Write-Host "Updated dataset config with version: $version"
    }
    
    # Process QDQ_model folder
    $qdqModelSource = Get-ChildItem -Path $tempPath -Recurse -Directory -Filter "QDQ_model" | Select-Object -First 1
    if ($qdqModelSource) {
        Write-Host "Processing QDQ_model..."
        $modelTargetPath = "./models/$modelName"
        New-Item -ItemType Directory -Path $modelTargetPath -Force | Out-Null
        
        # Find .onnx files in QDQ_model folder
        $onnxFiles = Get-ChildItem -Path $qdqModelSource.FullName -Filter "*.onnx" -Recurse
        foreach ($onnxFile in $onnxFiles) {
            $newModelName = "$modelName.quant.onnx"
            $destinationPath = Join-Path $modelTargetPath $newModelName
            Copy-Item -Path $onnxFile.FullName -Destination $destinationPath -Force
            Write-Host "  Copied and renamed: $($onnxFile.Name) -> $newModelName"
        }
        
        # Also copy .onnx.data files if present
        $onnxDataFiles = Get-ChildItem -Path $qdqModelSource.FullName -Filter "*.onnx.data" -Recurse
        foreach ($dataFile in $onnxDataFiles) {
            $destinationPath = Join-Path $modelTargetPath $dataFile.Name
            Copy-Item -Path $dataFile.FullName -Destination $destinationPath -Force
            Write-Host "  Copied: $($dataFile.Name)"
        }
    }
    
    # Convert raw files to npy after organizing
    Process-DatasetConversion -dataPath $dataPath -modelName $modelName -version $version
}

function Get-UPack {
    param (
        [string]$feed,
        [string]$name,
        [string]$version,
        [string]$modelName,
        [bool]$isCombinedPackage = $false
    )

    # Check if the data directory already exists
    $dataPath = "./data/$modelName"
    if (Test-Path -Path $dataPath) {
        Write-Host "`nData directory $dataPath already exists. Skipping download.`n"
        return
    }

    # Create temp directory for download
    $tempPath = "./temp_$name"
    
    # Download the package
    Download-UPack -feed $feed -name $name -version $version -outputPath $tempPath
    
    if ($isCombinedPackage) {
        # New approach: Process combined package
        Process-CombinedPackage -tempPath $tempPath -dataPath $dataPath -modelName $modelName -version $version
    } else {
        # Old approach: Separate dataset package
        # Create data directory structure with version subfolders inside input_data and output_data
        New-Item -ItemType Directory -Path "$dataPath" -Force | Out-Null
        New-Item -ItemType Directory -Path "$dataPath/input_data/$version" -Force | Out-Null
        New-Item -ItemType Directory -Path "$dataPath/output_data/$version" -Force | Out-Null
        
        # Copy dataset config
        $datasetConfigSource = "./dataset_configs/$modelName.json"
        if (Test-Path -Path $datasetConfigSource) {
            Copy-Item -Path $datasetConfigSource -Destination "$dataPath/dataset_config.json" -Force
            Write-Host "Copied dataset config for $modelName"
        } else {
            Write-Warning "Dataset config not found: $datasetConfigSource"
        }
        
        # Copy Model_<modelName>.json if it exists in the downloaded package
        $modelConfigSource = Get-ChildItem -Path $tempPath -Recurse -Filter "Model_$($modelName.ToUpper()).json" | Select-Object -First 1
        if ($modelConfigSource) {
            Copy-Item -Path $modelConfigSource.FullName -Destination "$dataPath/Model_$($modelName.ToUpper()).json" -Force
            Write-Host "Copied model config: Model_$($modelName.ToUpper()).json"
        } else {
            Write-Warning "Model config not found in package: Model_$($modelName.ToUpper()).json"
        }
        
        # Find and copy data folders
        $folders = Get-ChildItem -Path $tempPath -Directory
        foreach ($folder in $folders) {
            if ($folder.Name -like "*out*") {
                # Copy contents to output_data version folder
                Copy-Item -Path "$($folder.FullName)/*" -Destination "$dataPath/output_data/$version/" -Recurse -Force
                Write-Host "Copied output data from $($folder.Name)"
            } else {
                # Copy contents to input_data version folder
                Copy-Item -Path "$($folder.FullName)/*" -Destination "$dataPath/input_data/$version/" -Recurse -Force
                Write-Host "Copied input data from $($folder.Name)"
            }
        }
        
        # Organize files based on dataset config
        Organize-DatasetFiles -dataPath $dataPath -modelName $modelName -version $version
    }
    
    # Clean up temp directory
    Remove-Item -Path $tempPath -Recurse -Force
}

function Get-NuGet {
    param (
        [string]$feed,
        [string]$name,
        [string]$version,
        [string]$modelName
    )
    
    # Check if the model already exists
    $modelPath = "./models/$modelName"
    if (Test-Path -Path $modelPath) {
        $modelFiles = Get-ChildItem -Path $modelPath -Include *.onnxe, *.onnx
        if ($modelFiles.Count -gt 0) {
            Write-Host "`nModel $modelName already exists. Skipping download.`n"
            return
        }
    }
    
    # Define the command and arguments
    $command = "nuget.exe"
    $outputPath = "./temp_nuget_$modelName"
    $arguments = @(
        "install", "$name",
        "-Source", "https://devicesasg.pkgs.visualstudio.com/PerceptiveShell/_packaging/$feed/nuget/v3/index.json",
        "-Version", "$version",
        "-o", "$outputPath",
        "-DependencyVersion", "Ignore",
        "-Verbosity", "detailed"
    )
    Write-Host "`nDownloading NuGet model package:`n  Name: $name`n  Version: $version`n  Model: $modelName`n"
    & $command $arguments
    Move-Models -outputPath $outputPath -modelName $modelName
}

# Create base directories
New-Item -ItemType Directory -Path "./data" -Force | Out-Null
New-Item -ItemType Directory -Path "./models" -Force | Out-Null

# Read and parse JSON file
$jsonContent = Get-Content -Path $configPath -Raw | ConvertFrom-Json

# Install Python packages from requirements.txt
Install-PythonRequirements

# Process each model
if ($jsonContent.models) {
    $modelNames = $jsonContent.models.PSObject.Properties.Name
    
    foreach ($modelName in $modelNames) {
        Write-Host "`n========== Processing model: $modelName =========="
        
        $modelConfig = $jsonContent.models.$modelName
        
        # Determine if this is a combined package (only dataset_package, no quantized_model)
        $isCombinedPackage = ($modelConfig.dataset_package -and -not $modelConfig.quantized_model)
        
        # Download dataset package if exists
        if ($modelConfig.dataset_package) {
            $datasetPackage = $modelConfig.dataset_package
            Get-UPack -feed $datasetPackage.nuget_feed `
                      -name $datasetPackage.package_name `
                      -version $datasetPackage.package_version `
                      -modelName $modelName `
                      -isCombinedPackage $isCombinedPackage
        }
        
        # Download quantized model if exists (old approach)
        if ($modelConfig.quantized_model) {
            $quantizedModel = $modelConfig.quantized_model
            Get-NuGet -feed $quantizedModel.nuget_feed `
                      -name $quantizedModel.package_name `
                      -version $quantizedModel.package_version `
                      -modelName $modelName
        }
    }
} else {
    Write-Warning "No models found in configuration file"
}

Write-Host "`n========== Setup completed =========="
Write-Host "Directory structure created:"
Write-Host "  ./data/"
Get-ChildItem -Path "./data" -Directory | ForEach-Object { 
    $modelDir = $_
    Write-Host "    $($modelDir.Name)/"
    if (Test-Path "$($modelDir.FullName)/dataset_config.json") {
        Write-Host "      dataset_config.json"
    }
    @("input_data", "output_data", "l2_norm") | ForEach-Object {
        $subDir = $_
        $subDirPath = "$($modelDir.FullName)/$subDir"
        if (Test-Path $subDirPath) {
            Write-Host "      $subDir/"
            Get-ChildItem -Path $subDirPath -Directory | ForEach-Object {
                Write-Host "        $($_.Name)/"
            }
        }
    }
}
Write-Host "  ./models/"
Get-ChildItem -Path "./models" -Directory | ForEach-Object { Write-Host "    $($_.Name)/" }