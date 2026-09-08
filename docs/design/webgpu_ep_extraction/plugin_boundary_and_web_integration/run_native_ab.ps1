# Interleaved A/B: built-in WebGPU EP vs shared-library plugin EP.
# Parses "Min Latency" from stdout; perf_test's exit code is unreliable because the
# shutdown leak checker reports benign CRT static-init allocations.
param(
  [int]$Rounds = 8,
  [int]$Reps = 30,
  [int]$QwenReps = 20,
  [string]$OutCsv = 'D:\test\native_ab_raw.csv'
)

$builtinExe = 'D:\source\onnxruntime_4\build\nat_builtin\Release\onnxruntime_perf_test.exe'
$pluginDir  = 'D:\source\onnxruntime_4\build\nat_plugin\Release'
$pluginExe  = "$pluginDir\onnxruntime_perf_test.exe"

$models = [ordered]@{
  'bench_compute'  = 'D:\source\onnxruntime_4\js\web\test\data\bench\bench_compute\model.onnx'
  'bench_dispatch' = 'D:\source\onnxruntime_4\js\web\test\data\bench\bench_dispatch\model.onnx'
  'qwen_decode'    = 'D:\test\qwen_perf\decode\model.onnx'
  'qwen_prefill'   = 'D:\test\qwen_perf\prefill\model.onnx'
}

function Get-MinLatency([string]$text) {
  $m = [regex]::Match($text, 'Min Latency:\s*([0-9.eE+-]+)\s*s')
  if ($m.Success) { return [double]$m.Groups[1].Value }
  return $null
}

function Invoke-Arm([string]$arm, [string]$model, [int]$reps) {
  if ($arm -eq 'builtin') {
    $out = & $builtinExe -e webgpu -m times -r $reps $model 2>&1 | Out-String
  } else {
    Push-Location $pluginDir
    try {
      $out = & $pluginExe --plugin_ep_libs "WebGPU|onnxruntime_providers_webgpu.dll" `
                          --plugin_eps WebGpuExecutionProvider -m times -r $reps $model 2>&1 | Out-String
    } finally { Pop-Location }
  }
  return (Get-MinLatency $out)
}

function Get-Median([double[]]$v) {
  $s = $v | Sort-Object
  $n = $s.Count
  if ($n -eq 0) { return $null }
  if ($n % 2 -eq 1) { return $s[[int](($n-1)/2)] }
  return ($s[$n/2 - 1] + $s[$n/2]) / 2
}

$rows = @()
foreach ($r in 1..$Rounds) {
  foreach ($name in $models.Keys) {
    $reps = if ($name -like 'qwen*') { $QwenReps } else { $Reps }
    # Interleave within the round so slow drift affects both arms equally.
    $b = Invoke-Arm 'builtin' $models[$name] $reps
    $p = Invoke-Arm 'plugin'  $models[$name] $reps
    $rows += [pscustomobject]@{
      round = $r; model = $name
      builtin_ms = if ($b -ne $null) { [math]::Round($b*1000,4) } else { $null }
      plugin_ms  = if ($p -ne $null) { [math]::Round($p*1000,4) } else { $null }
    }
    Write-Host ("round {0} {1,-15} builtin={2,9:N4} ms  plugin={3,9:N4} ms" -f $r,$name,($b*1000),($p*1000))
  }
}

$rows | Export-Csv -NoTypeInformation -Path $OutCsv
Write-Host "`n=== raw rows written to $OutCsv ==="
Write-Host "`n=== SUMMARY (median of per-round minimums) ==="
Write-Host ("{0,-15} {1,12} {2,12} {3,10} {4,9} {5,13}" -f `
  'model','builtin_ms','plugin_ms','delta_ms','ratio','plugin_slower')
foreach ($name in $models.Keys) {
  $sub = $rows | Where-Object { $_.model -eq $name -and $_.builtin_ms -ne $null -and $_.plugin_ms -ne $null }
  if (-not $sub) { Write-Host ("{0,-15} NO DATA" -f $name); continue }
  $mb = Get-Median ([double[]]($sub.builtin_ms))
  $mp = Get-Median ([double[]]($sub.plugin_ms))
  $slower = ($sub | Where-Object { $_.plugin_ms -gt $_.builtin_ms }).Count
  Write-Host ("{0,-15} {1,12:N4} {2,12:N4} {3,10:N4} {4,9:N3} {5,10}/{6}" -f `
    $name, $mb, $mp, ($mp-$mb), $(if($mb -gt 0){$mp/$mb}else{0}), $slower, $sub.Count)
}
