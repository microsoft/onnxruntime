param(
    [string]$instruction, # Should be 'extract' or 'repack'
    [string]$jar_file_directory # The directory where the original jar file is located
)

$extracted_file_directory = Join-Path $jar_file_directory "jar_extracted_full_files"
$state_file = Join-Path $jar_file_directory "repack_list.txt"

if ($instruction -eq "extract") {
    # Find the main jar file(s) by looking for names that start with 'onnxruntime'
    # and excluding common suffixes for sources and javadocs.
    $main_jar_files = Get-ChildItem -Path $jar_file_directory -Filter onnxruntime*.jar | Where-Object { $_.Name -notlike '*-sources.jar' -and $_.Name -notlike '*-javadoc.jar' }

    if ($main_jar_files.Count -eq 0) {
        Write-Error "No main ONNX Runtime JAR file found in directory: $jar_file_directory"
        exit 1
    }

    # Clear any previous state file
    if (Test-Path $state_file) {
        Remove-Item $state_file
    }

    foreach ($jar_file in $main_jar_files) {
        Write-Host "Extracting the jar file $($jar_file.FullName)..."
        & 7z x $jar_file.FullName -o"$extracted_file_directory"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "7z failed to extract the jar file. Exitcode: $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        
        # Save the original name for repacking, then remove the file
        $jar_file.Name | Out-File -FilePath $state_file -Append
        Write-Host "Removing the original jar file: $($jar_file.FullName)"
        Remove-Item -Path $jar_file.FullName -Force
    }
    Write-Host "Extracted files to directory: $extracted_file_directory"

} elseif ($instruction -eq "repack") {
    if (-not (Test-Path $state_file)) {
        Write-Error "State file '$state_file' not found. Cannot repack."
        exit 1
    }
    
    Write-Host "Removing ESRP's CodeSignSummary file..."
    Remove-Item -Path "$extracted_file_directory/CodeSignSummary*.*" -Force -ErrorAction SilentlyContinue
    Write-Host "Removed ESRP's CodeSignSummary file."

    $jar_files_to_repack = Get-Content $state_file

    foreach ($jar_file_name in $jar_files_to_repack) {
        $repacked_jar_file_path = Join-Path $jar_file_directory $jar_file_name
        Write-Host "Repacking to $repacked_jar_file_path from directory $extracted_file_directory..."
        & 7z a "$repacked_jar_file_path" "$extracted_file_directory\*"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "7z failed to repack the jar file. Exitcode: $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        Write-Host "Repacked the jar file $repacked_jar_file_path."
    }

    Write-Host "Removing the extracted files and state file..."
    Remove-Item -Path "$extracted_file_directory" -Recurse -Force
    Remove-Item -Path $state_file -Force
    Write-Host "Cleaned up temporary files."

} else {
    Write-Error "Invalid instruction: '$instruction'. Must be 'extract' or 'repack'."
    exit 1
}
