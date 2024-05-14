# The first part of the expression in the following "if" statement is necessary, otherwise it will report an error,
# "You cannot call a method on a null-valued expression.", when the env variable does not exist.
if (-not [string]::IsNullOrEmpty( $Env:TELEMETRYGUID) -and $Env:TELEMETRYGUID.StartsWith('"')) {
    $length = $Env:TELEMETRYGUID.length
    # See https://learn.microsoft.com/en-us/windows/win32/api/traceloggingprovider/nf-traceloggingprovider-traceloggingoptiongroup
    # The value of the env variable must have quotes, because we do not add the quotes here.
    $fileContent = "#define TraceLoggingOptionMicrosoftTelemetry() \
      TraceLoggingOptionGroup("+$Env:TELEMETRYGUID.substring(1, $length-2)+")"
    New-Item -Path "include\onnxruntime\core\platform\windows\TraceLoggingConfigPrivate.h" -ItemType "file" -Value "$fileContent" -Force
    Write-Host "##vso[task.setvariable variable=TelemetryOption]--use_telemetry"
    Write-Host "Telemetry is enabled."
} else {
    Write-Host "##vso[task.setvariable variable=TelemetryOption]"
    Write-Host "Telemetry is disabled."
}