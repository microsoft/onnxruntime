# ORT Perf Test

## build

1. Configure based on arch

`chrisd\p-x64.cmd`
or
`chrisd\p-arm.cmd`

2. Build

`chrisd\b.cmd`

## Run Tests

### QNN
`chrisd\go-qnn.cmd`

### OpenVino
`chrisd\go-openvino.cmd`

### NVidia
`chrisd\go-nvidia-tests.cmd`

## HowTo - Determine what WinAppSDK Runtimes are installed.

```pwsh
Get-AppxPackage -Name "Microsoft.WindowsAppRuntime.*" | Select-Object -ExpandProperty PackageFamilyName -Unique | Sort-Object -Descending
```

## Appendix

[WinAppSDK - 20-experimental](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads#windows-app-sdk-20-experimental)

[Calling Style](https://dev.azure.com/WSSI/COMPUTE/_git/WCR.Cert?path=/evaluator/asg/asg_model_evaluator.py&version=GBmain&_a=contents)
