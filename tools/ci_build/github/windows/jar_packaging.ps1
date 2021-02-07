$ErrorActionPreference = "Stop"
Write-Output "Start"
dir
pushd onnxruntime-java-linux-x64
Write-Output "Run 7z"
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\testing.jar libcustom_op_library.so
Remove-Item -Path libcustom_op_library.so
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\onnxruntime-$Env:ONNXRUNTIMEVERSION.jar .
popd
pushd onnxruntime-java-osx-x64
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\testing.jar libcustom_op_library.dylib
Remove-Item -Path libcustom_op_library.dylib
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\onnxruntime-$Env:ONNXRUNTIMEVERSION.jar .
popd