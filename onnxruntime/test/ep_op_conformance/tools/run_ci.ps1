[CmdletBinding()]
param(
    [string]$BundleDirectory = $PSScriptRoot,
    [string]$ExpectedRuntimeVersion = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$bundle = (Resolve-Path -LiteralPath $BundleDirectory).Path
$reports = Join-Path $bundle 'reports'
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$checks = [System.Collections.Generic.List[object]]::new()
$originalPath = $env:PATH
$originalPythonPath = $env:PYTHONPATH
$originalPythonHome = $env:PYTHONHOME

function Assert-Condition([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        throw $Message
    }
}

function Invoke-Scenario {
    param(
        [string]$Name,
        [string]$Cases = 'positive',
        [string]$Profile = 'cpu',
        [int]$ExitCode = 0,
        [string]$Status = 'PASS',
        [int]$Passed = 0,
        [int]$Failed = 0,
        [int]$Skipped = 0,
        [string]$ErrorPattern = '',
        [string]$Phase = ''
    )
    $process = [System.Diagnostics.Process]::new()
    try {
        $reportPath = Join-Path $reports "$Name.json"
        if (Test-Path -LiteralPath $reportPath) {
            Remove-Item -LiteralPath $reportPath
        }

        $process.StartInfo.FileName = Join-Path $bundle 'bin/onnxruntime_ep_op_conformance_test.exe'
        $process.StartInfo.WorkingDirectory = $bundle
        $process.StartInfo.UseShellExecute = $false
        $process.StartInfo.RedirectStandardOutput = $true
        $process.StartInfo.RedirectStandardError = $true
        foreach ($argument in @(
            '--runtime', (Join-Path $bundle 'bin/onnxruntime.dll'),
            '--profile', (Join-Path $bundle "profiles/$Profile.json"),
            '--cases', (Join-Path $bundle "cases/$Cases"),
            '--report', $reportPath
        )) {
            $process.StartInfo.ArgumentList.Add($argument)
        }
        Assert-Condition $process.Start() "$Name could not start"
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(120000)) {
            $process.Kill($true)
            $process.WaitForExit()
            throw "$Name timed out"
        }
        $stdout.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $reports "$Name.stdout.log")
        $stderr.GetAwaiter().GetResult() | Set-Content -LiteralPath (Join-Path $reports "$Name.stderr.log")
        Assert-Condition ($process.ExitCode -eq $ExitCode) "$Name returned $($process.ExitCode), expected $ExitCode"
        $report = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json -AsHashtable
        Assert-Condition ($report.schema_version -eq 1) "$Name report schema mismatch"
        Assert-Condition ($report.status -ceq $Status) "$Name status $($report.status), expected $Status"
        if ($ExpectedRuntimeVersion -and $report.ContainsKey('runtime_version')) {
            Assert-Condition ($report.runtime_version -ceq $ExpectedRuntimeVersion) "$Name runtime version mismatch"
        }
        if ($Status -eq 'ERROR') {
            Assert-Condition ($report.error -match $ErrorPattern) "$Name did not report the expected setup error"
            Assert-Condition (@($report.results).Count -eq 0) "$Name unexpectedly executed cases"
        } else {
            Assert-Condition ($report.profile.kind -ceq 'builtin') "$Name target kind mismatch"
            Assert-Condition ($report.profile.ep_name -ceq 'CPUExecutionProvider') "$Name target name mismatch"
            Assert-Condition ($report.passed -eq $Passed) "$Name pass count mismatch"
            Assert-Condition ($report.failed -eq $Failed) "$Name fail count mismatch"
            Assert-Condition ($report.skipped -eq $Skipped) "$Name skip count mismatch"
            $expectedIds = @(
                if ($Cases -eq 'positive') {
                    $metadata.positive_case_ids
                } else {
                    $metadata.mutation_case_id
                }
            )
            $actualIds = @($report.results | ForEach-Object { $_.id })
            Assert-Condition ($actualIds.Count -eq $expectedIds.Count) "$Name case count mismatch"
            Assert-Condition (@(Compare-Object $expectedIds $actualIds).Count -eq 0) "$Name case identity mismatch"
            $expectedCounts = @{ PASS = $Passed; FAIL = $Failed; SKIP = $Skipped }
            foreach ($terminal in $expectedCounts.Keys) {
                $actualCount = @($report.results | Where-Object { $_.status -ceq $terminal }).Count
                Assert-Condition ($actualCount -eq $expectedCounts[$terminal]) "$Name $terminal result count mismatch"
            }
            foreach ($result in $report.results) {
                if ($result.status -eq 'SKIP') {
                    Assert-Condition (-not [string]::IsNullOrWhiteSpace($result.reason)) "$Name skip has no reason"
                    continue
                }
                Assert-Condition (@($result.assignment).Count -gt 0) "$Name has no assignment evidence"
                Assert-Condition (@($result.assignment | Where-Object { $_.node -ceq 'node1' }).Count -eq 1) "$Name required node is missing"
                foreach ($assignment in $result.assignment) {
                    Assert-Condition ($assignment.ep_name -ceq 'CPUExecutionProvider') "$Name assigned to the wrong EP"
                }
                if ($result.status -eq 'FAIL') {
                    Assert-Condition ($result.phase -ceq $Phase) "$Name failed in the wrong phase"
                    Assert-Condition (-not [string]::IsNullOrWhiteSpace($result.reason)) "$Name failure has no reason"
                    if ($ErrorPattern) {
                        Assert-Condition ($result.reason -match $ErrorPattern) "$Name failure diagnostic mismatch"
                    }
                }
            }
        }
        $checks.Add(@{ name = $Name; status = 'PASS'; exit_code = $process.ExitCode })
        Write-Host "$Name validated (exit $ExitCode, $Status)"
    } catch {
        $checks.Add(@{ name = $Name; status = 'FAIL'; error = $_.Exception.Message })
        Write-Host "${Name}: $($_.Exception.Message)"
    } finally {
        $process.Dispose()
    }
}

try {
    $env:PATH = Join-Path $env:SYSTEMROOT 'System32'
    $env:PYTHONPATH = $null
    $env:PYTHONHOME = $null
    foreach ($command in @('python', 'python3', 'py', 'git', 'cmake', 'cl')) {
        Assert-Condition (-not (Get-Command $command -ErrorAction SilentlyContinue)) "$command is unexpectedly discoverable"
    }
    $metadata = Get-Content -Raw -LiteralPath (Join-Path $bundle 'prepared.json') | ConvertFrom-Json -AsHashtable
    Assert-Condition ($metadata.schema_version -eq 1) 'Unsupported prepared bundle schema'
    Assert-Condition (@($metadata.positive_case_ids).Count -gt 1) 'Expected multiple canonical positive cases'
    foreach ($entry in $metadata.files.GetEnumerator()) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $bundle $entry.Key)).Hash
        Assert-Condition ($hash -ieq $entry.Value) "Bundle hash mismatch: $($entry.Key)"
    }
    $total = @($metadata.positive_case_ids).Count
    Invoke-Scenario -Name 'positive' -Passed $total
    Invoke-Scenario -Name 'wrong-output' -Cases 'wrong-output' -ExitCode 1 -Status 'FAIL' -Failed 1 `
        -Phase 'execution_or_comparison' -ErrorPattern '^Output mismatch at element '
    Invoke-Scenario -Name 'malformed-artifact' -Cases 'malformed-artifact' -ExitCode 2 -Status 'ERROR' -ErrorPattern 'Invalid TensorProto'
    Invoke-Scenario -Name 'invalid-profile' -Profile 'invalid-profile' -ExitCode 2 -Status 'ERROR' -ErrorPattern 'Unknown target kind'
    Invoke-Scenario -Name 'unknown-case-field' -Cases 'unknown-case-field' -ExitCode 2 -Status 'ERROR' -ErrorPattern 'Unsupported field: ignored_assertion'
    Invoke-Scenario -Name 'unknown-profile-field' -Profile 'unknown-profile-field' -ExitCode 2 -Status 'ERROR' -ErrorPattern 'Unsupported field: provider_name'
    Invoke-Scenario -Name 'partial-skip' -Profile 'partial-skip' -Passed ($total - 1) -Skipped 1
    Invoke-Scenario -Name 'all-skipped' -Profile 'all-skipped' -ExitCode 1 -Status 'FAIL' -Skipped $total
    Invoke-Scenario -Name 'unknown-skip' -Profile 'unknown-skip' -ExitCode 2 -Status 'ERROR' -ErrorPattern 'Skip must name'
} catch {
    $checks.Add(@{ name = 'bundle-preflight'; status = 'FAIL'; error = $_.Exception.Message })
} finally {
    $env:PATH = $originalPath
    $env:PYTHONPATH = $originalPythonPath
    $env:PYTHONHOME = $originalPythonHome
    $failures = @($checks | Where-Object { $_.status -eq 'FAIL' }).Count
    $summaryStatus = if ($failures) {
        'FAIL'
    } else {
        'PASS'
    }

    @{
        schema_version = 1
        status = $summaryStatus
        checks = @($checks.ToArray())
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $reports 'ci-summary.json')
}
if ($failures) {
    throw "$failures CPU CI checks failed; see $reports"
}

Write-Host "All $($checks.Count) native CPU CI checks passed without Python or source tools on PATH."