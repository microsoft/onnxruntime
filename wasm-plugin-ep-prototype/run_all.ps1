# Build and run the whole prototype matrix, in Node and in a headless browser.
#
#   .\run_all.ps1
#
# Requires: ..\emsdk (Emscripten 4.0.23) and ..\pw-browsers (Playwright chromium).

$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$emsdkEnv = Join-Path (Split-Path $root -Parent) 'emsdk\emsdk_env.bat'
$node = Join-Path (Split-Path $root -Parent) 'emsdk\node\22.16.0_64bit\bin\node.exe'
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path (Split-Path $root -Parent) 'pw-browsers'

# name                                     build args                                   node modes            browser modes
$matrix = @(
    @{ Name = 'baseline (no dynamic linking)'; Args = @('-MainModule', '0'); Node = @(); Browser = @() },
    @{ Name = 'MAIN_MODULE=1, legacy EH'; Args = @('-MainModule', '1'); Node = @('preload', 'dlopen'); Browser = @('preload') },
    @{ Name = 'MAIN_MODULE=1, legacy EH, ASYNCIFY'; Args = @('-MainModule', '1', '-Asyncify'); Node = @('preload', 'ondemand'); Browser = @('ondemand') },
    @{ Name = 'MAIN_MODULE=2, legacy EH'; Args = @('-MainModule', '2'); Node = @('preload'); Browser = @() },
    @{ Name = 'MAIN_MODULE=2, wasm EH'; Args = @('-MainModule', '2', '-WasmEH'); Node = @('preload', 'dlopen'); Browser = @('preload', 'dlopen') },
    @{ Name = 'MAIN_MODULE=2, wasm EH, pthreads'; Args = @('-MainModule', '2', '-WasmEH', '-Pthreads'); Node = @('preload'); Browser = @('preload') },
    @{ Name = 'MAIN_MODULE=2, wasm EH, pthreads, JSPI, >4KB EP'; Args = @('-MainModule', '2', '-WasmEH', '-Pthreads', '-Jspi', '-Pad'); Node = @(); Browser = @('preload', 'ondemand') }
)

$results = @()

foreach ($cfg in $matrix) {
    Write-Host "`n##################################################" -ForegroundColor Green
    Write-Host "## $($cfg.Name)" -ForegroundColor Green
    Write-Host "##################################################" -ForegroundColor Green

    $buildArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $root 'build.ps1')) + $cfg.Args
    $buildLog = & cmd /c "call `"$emsdkEnv`" >nul 2>&1 && powershell $($buildArgs -join ' ')" 2>&1
    $buildOk = $LASTEXITCODE -eq 0
    $buildLog | Select-String 'auto-derived|^ort_host|^plugin_ep|error' | ForEach-Object { Write-Host "  $_" }

    # Recover the output directory name the same way build.ps1 derives it.
    $dir = "build-mm$($cfg.Args[1])"
    if ($cfg.Args -contains '-WasmEH') { $dir += '-wasmeh' }
    if ($cfg.Args -contains '-Pthreads') { $dir += '-pthreads' }
    if ($cfg.Args -contains '-Asyncify') { $dir += '-asyncify' }
    if ($cfg.Args -contains '-Jspi') { $dir += '-jspi' }
    if ($cfg.Args -contains '-Pad') { $dir += '-pad' }

    $hostWasm = Join-Path $root "$dir\ort_host.wasm"
    $epWasm = Join-Path $root "$dir\plugin_ep.wasm"
    $hostKB = if (Test-Path $hostWasm) { [math]::Round((Get-Item $hostWasm).Length / 1KB, 1) } else { 'n/a' }
    $epKB = if (Test-Path $epWasm) { [math]::Round((Get-Item $epWasm).Length / 1KB, 1) } else { 'n/a' }

    $nodeResult = if ($cfg.Node.Count -eq 0) { '-' } else { 'PASS' }
    foreach ($m in $cfg.Node) {
        Write-Host "  -- node ($m)" -ForegroundColor Cyan
        $out = & cmd /c "call `"$emsdkEnv`" >nul 2>&1 && `"$node`" `"$(Join-Path $root 'run.js')`" `"$(Join-Path $root $dir)`" $m" 2>&1
        if ($LASTEXITCODE -ne 0) { $nodeResult = "FAIL($m)"; $out | Select-Object -Last 4 | ForEach-Object { Write-Host "     $_" } }
    }

    $browserResult = if ($cfg.Browser.Count -eq 0) { '-' } else { 'PASS' }
    if ($cfg.Browser.Count -gt 0) {
        Write-Host "  -- browser ($($cfg.Browser -join ','))" -ForegroundColor Cyan
        $out = & $node (Join-Path $root 'browser_run.js') (Join-Path $root $dir) ($cfg.Browser -join ',') 2>&1
        if ($LASTEXITCODE -ne 0) { $browserResult = 'FAIL' }
        $out | Select-String 'STATUS=|sync compile|chromium' | ForEach-Object { Write-Host "     $_" }
    }

    $results += [pscustomobject]@{
        Config = $cfg.Name
        Build  = if ($buildOk) { 'ok' } else { 'FAILED' }
        HostKB = $hostKB
        EpKB   = $epKB
        Node   = $nodeResult
        Chrome = $browserResult
    }
}

Write-Host "`n================ SUMMARY ================" -ForegroundColor Green
$results | Format-Table -AutoSize
