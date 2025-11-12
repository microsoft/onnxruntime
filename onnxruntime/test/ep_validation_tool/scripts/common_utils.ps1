# ----------------------------------------
# Function Definitions
# ----------------------------------------

# Function to load JSON config
function Load-JsonConfig {
    param (
        [string]$configPath
    )
    $jsonContent = Get-Content -Raw -Path $configPath | ConvertFrom-Json
    return $jsonContent
}

# Function to convert JSON config to command-line arguments
function Convert-JsonToArgs {
    param (
        [PSCustomObject]$config
    )
    $args = @()

    $args += "--executionProvider"
    $args += $executionProvider

    if ($config.arguments)
    {
        foreach ($argument in $config.arguments.PSObject.Properties) {
            $args += "--$($argument.Name)"
            $args += $argument.Value
        }
    }

    if ($config.ep_specific_arguments.$executionProvider)
    {
        foreach ($argument in $config.ep_specific_arguments.$executionProvider.PSObject.Properties) {
            $args += "--$($argument.Name)"
            $args += $argument.Value
        }
    }

    return $args
}

function Format-Json {
    <#
    .SYNOPSIS
        Prettifies JSON output.
        Version January 3rd 2024
        Fixes:
            - empty [] or {} or in-line arrays as per https://stackoverflow.com/a/71664664/9898643
              by Widlov (https://stackoverflow.com/users/1716283/widlov)
            - Unicode Apostrophs \u0027 as written by ConvertTo-Json are replaced with regular single quotes "'"
            - multiline empty [] or {} are converted into inline arrays or objects
    .DESCRIPTION
        Reformats a JSON string so the output looks better than what ConvertTo-Json outputs.
    .PARAMETER Json
        Required: [string] The JSON text to prettify.
    .PARAMETER Minify
        Optional: Returns the json string compressed.
    .PARAMETER Indentation
        Optional: The number of spaces (1..1024) to use for indentation. Defaults to 2.
    .PARAMETER AsArray
        Optional: If set, the output will be in the form of a string array, otherwise a single string is output.
    .EXAMPLE
        $json | ConvertTo-Json | Format-Json -Indentation 4
    .OUTPUTS
        System.String or System.String[] (the latter when parameter AsArray is set)
    #>
    [CmdletBinding(DefaultParameterSetName = 'Prettify')]
    Param(
        [Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
        [string]$Json,

        [Parameter(ParameterSetName = 'Minify')]
        [switch]$Minify,

        [Parameter(ParameterSetName = 'Prettify')]
        [ValidateRange(1, 1024)]
        [int]$Indentation = 2,

        [Parameter(ParameterSetName = 'Prettify')]
        [switch]$AsArray
    )

    if ($PSCmdlet.ParameterSetName -eq 'Minify') {
        return ($Json | ConvertFrom-Json) | ConvertTo-Json -Depth 100 -Compress
    }

    # If the input JSON text has been created with ConvertTo-Json -Compress
    # then we first need to reconvert it without compression
    if ($Json -notmatch '\r?\n') {
        $Json = ($Json | ConvertFrom-Json) | ConvertTo-Json -Depth 100
    }

    $indent = 0
    $regexUnlessQuoted = '(?=([^"]*"[^"]*")*[^"]*$)'

    $result = ($Json -split '\r?\n' | ForEach-Object {
        # If the line contains a ] or } character, 
        # we need to decrement the indentation level unless:
        #   - it is inside quotes, AND
        #   - it does not contain a [ or {
        if (($_ -match "[}\]]$regexUnlessQuoted") -and ($_ -notmatch "[\{\[]$regexUnlessQuoted")) {
            $indent = [Math]::Max($indent - $Indentation, 0)
        }

        # Replace all colon-space combinations by ": " unless it is inside quotes.
        $line = (' ' * $indent) + ($_.TrimStart() -replace ":\s+$regexUnlessQuoted", ': ')

        # If the line contains a [ or { character, 
        # we need to increment the indentation level unless:
        #   - it is inside quotes, AND
        #   - it does not contain a ] or }
        if (($_ -match "[\{\[]$regexUnlessQuoted") -and ($_ -notmatch "[}\]]$regexUnlessQuoted")) {
            $indent += $Indentation
        }

        # ConvertTo-Json returns all single-quote characters as Unicode Apostrophs \u0027
        # see: https://stackoverflow.com/a/29312389/9898643
        $line -replace '\\u0027', "'"

    # join the array with newlines and convert multiline empty [] or {} into inline arrays or objects
    }) -join [Environment]::NewLine -replace '(\[)\s+(\])', '$1$2' -replace '(\{)\s+(\})', '$1$2'

    if ($AsArray) { return ,[string[]]($result -split '\r?\n') }
    $result
}

# Function to gather device configuration information
function Get-DeviceConfig {
    param (
        [string]$exePath,
        [string]$executionProvider
    )
    
    # Get binaries path from exePath
    $binPath = Split-Path -Path $exePath -Parent
    $deviceConfig = [ordered] @{}
    
    # NPU information
    $validManufacturers = @("Qualcomm Technologies, Inc.", "Intel Corporation", "AMD")
    $driver = Get-WmiObject Win32_PnPSignedDriver | Where-Object {
        ($_.DeviceName -match "NPU" -or $_.DeviceName -match "AI Boost") -and $_.Manufacturer -in $validManufacturers
    } | Select-Object -First 1
    
    if ($driver) {
        $deviceConfig.manufacturer = $driver.Manufacturer
        $deviceConfig.device_name = $driver.DeviceName
        $deviceConfig.npu_driver_version = $driver.DriverVersion
        $deviceConfig.NPUFound = $true
    } else {
        $deviceConfig.manufacturer = "unknown"
        $deviceConfig.device_name = "unknown"
        $deviceConfig.npu_driver_version = "unknown"
        $deviceConfig.NPUFound = $false
    }
    
    # EP name
    $deviceConfig.execution_provider = $executionProvider
    
    # EP DLL version
    switch($executionProvider) {
        "qnn" {
            $dllName = "QnnHtp.dll"
        }
        "ov" {
            $dllName = "openvino.dll"
        }
        "vitisai" {
            $dllName = "onnxruntime_vitisai_ep.dll"
        }
        Default {
            throw "Invalid execution provider value: $executionProvider"
        }
    }
    
    $dllPath = "$binPath\$dllName"
    if (Test-Path $dllPath) {
        $deviceConfig.ep_dll_version = (Get-Item $dllPath).VersionInfo.FileVersionRaw.ToString()
        $deviceConfig.EPDllFound = $true
    } else {
        $deviceConfig.ep_dll_version = "unknown"
        $deviceConfig.EPDllFound = $false
    }
    
    # ORT version
    $ortPath = "$binPath\ps-onnxruntime.dll"
    if (Test-Path $ortPath) {
        $deviceConfig.ort_version = (Get-Item $ortPath).VersionInfo.FileVersionRaw.ToString()
        $deviceConfig.ORTFound = $true
    } else {
        $deviceConfig.ort_version = "unknown"
        $deviceConfig.ORTFound = $false
    }
    
    return [PSCustomObject]$deviceConfig
}

# Function to print device configuration to console
function Print-DeviceConfig {
    param (
        [PSCustomObject]$deviceConfig
    )
    
    Write-Host "----- Device configuration -----"
    
    if ($deviceConfig.NPUFound) {
        Write-Host "Manufacturer: $($deviceConfig.manufacturer)"
        Write-Host "Device Name: $($deviceConfig.device_name)"
        Write-Host "NPU Driver Version: $($deviceConfig.npu_driver_version)"
    } else {
        Write-Host "NPU driver not found for the specified manufacturers."
    }
    
    Write-Host "Execution provider: $($deviceConfig.execution_provider)"
    
    if ($deviceConfig.EPDllFound) {
        Write-Host "EP DLL Version: $($deviceConfig.ep_dll_version)"
    } else {
        Write-Host "DLL not found for the specified execution provider."
    }
    
    if ($deviceConfig.ORTFound) {
        Write-Host "ORT Version: $($deviceConfig.ort_version)`n"
    } else {
        Write-Host "ORT not found."
    }
}

# Function to dump device configuration to json file
function Dump-DeviceConfig {
    param (
        [PSCustomObject]$deviceConfig,
        [string]$outputPath
    )
    
    # Check if the output JSON already exists
    $jsonData = @{}
    if (Test-Path $outputPath) {
        try {
            $jsonData = Get-Content -Raw -Path $outputPath | ConvertFrom-Json
        } catch {
            Write-Warning "Could not read existing JSON file. Creating new one."
        }
    }

    # Remove all properties that contain "Found" in their name
    $propertiesToRemove = $deviceConfig.PSObject.Properties.Where({ $_.Name -like "*Found*" })
    foreach ($property in $propertiesToRemove) {
        $deviceConfig.PSObject.Properties.Remove($property.Name)
    }
    
    # Add device configuration to the JSON data
    $jsonData | Add-Member -MemberType NoteProperty -Name "device_configuration" -Value $deviceConfig -Force
    
    # Create the directory if it doesn't exist
    $outputDir = Split-Path -Path $outputPath -Parent
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Write to JSON file
    ConvertTo-Json @($jsonData) -Depth 10 | Format-Json | Set-Content $outputPath -Encoding UTF8
    Write-Host "Device configuration added to $outputPath"
}

function Get-DatasetDirectory {
    param (
        [string]$url
    )
    $datasetName = Split-Path -Path $datasetUrl -Leaf
    return Join-Path -Path "datasets" -ChildPath $datasetName
}
