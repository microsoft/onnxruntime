$ErrorActionPreference = "Stop"
Write-Output "Start"
dir
pushd onnxruntime-java-linux-x64
Write-Output "Run 7z"
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\testing.jar libcustom_op_library.so
Remove-Item -Path libcustom_op_library.so
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\onnxruntime-$Env:ONNXRUNTIMEVERSION.jar .
popd
pushd onnxruntime-java-win-x64
ren onnxruntime-$Env:ONNXRUNTIMEVERSION.jar onnxruntime_gpu-$Env:ONNXRUNTIMEVERSION.jar
ren onnxruntime-$Env:ONNXRUNTIMEVERSION-javadoc.jar onnxruntime_gpu-$Env:ONNXRUNTIMEVERSION-javadoc.jar
ren onnxruntime-$Env:ONNXRUNTIMEVERSION-sources.jar onnxruntime_gpu-$Env:ONNXRUNTIMEVERSION-sources.jar
ren onnxruntime-$Env:ONNXRUNTIMEVERSION.pom onnxruntime_gpu-$Env:ONNXRUNTIMEVERSION.pom
popd