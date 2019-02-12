# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#Copy CUDA props files
Param(
    [Parameter(Mandatory=$True)]
    [string]$CudaMsbuildPath,
    [string]$CudaVersion
)
$CudaMsbuildPath=$(Join-Path $CudaMsbuildPath "V141\BuildCustomizations")
$Dst = $(Join-Path $env:VS2017INSTALLDIR "Common7\IDE\VC\VCTargets\BuildCustomizations")

Write-Host "Copy CUDA prop files"
Copy-Item  $(Join-Path $CudaMsbuildPath "CUDA ${CudaVersion}.props")  -Destination  $Dst
Copy-Item  $(Join-Path $CudaMsbuildPath "CUDA ${CudaVersion}.targets") -Destination  $Dst
Copy-Item  $(Join-Path $CudaMsbuildPath "CUDA ${CudaVersion}.xml")  -Destination  $Dst
Copy-Item  $(Join-Path $CudaMsbuildPath "Nvda.Build.CudaTasks.v${CudaVersion}.dll") -Destination  $Dst

