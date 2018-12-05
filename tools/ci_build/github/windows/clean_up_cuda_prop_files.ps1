# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#Clean up CUDA props files
Param(
    [Parameter(Mandatory=$True)]
    [string]$CudaVersion
)
$Dst = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations"

Write-Host "Clean up CUDA prop files"
Remove-Item  $(Join-Path $Dst "CUDA ${CudaVersion}.props")  
Remove-Item  $(Join-Path $Dst "CUDA ${CudaVersion}.targets") 
Remove-Item  $(Join-Path $Dst "CUDA ${CudaVersion}.xml")  
Remove-Item  $(Join-Path $Dst "Nvda.Build.CudaTasks.v${CudaVersion}.dll") 

