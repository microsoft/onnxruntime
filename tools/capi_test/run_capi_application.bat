set ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
set ONNX_MODEL="squeezenet.onnx"
SET ORT_PACKAGE=%1
SET WORKSPACE=%2

echo The current directory is %CD%

7z.exe x %ORT_PACKAGE% -y
set ORT_LIB=%ORT_PACKAGE:~0,-4%\lib
echo %ORT_LIB%

cd %WORKSPACE%
cmake.exe -S . -B build\ -G "Visual Studio 16 2019"

REM Copy ORT libraries to same folder for linker to build.
REM For some reasons, setting "LINK" or "LIBPATH" env variables won't help. 
cd build
powershell -Command "cp %ORT_LIB%\* ."
MSBuild.exe .\capi_test.sln /property:Configuration=Release

REM Copy ORT libraries to same folder for executable to run.
cd Release
powershell -Command "cp %ORT_LIB%\* ."
powershell -Command "Invoke-WebRequest %ONNX_MODEL_URL% -Outfile %ONNX_MODEL%"
capi_test.exe
