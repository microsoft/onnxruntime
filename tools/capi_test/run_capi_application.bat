set ONNX_MODEL_URL="https://media.githubusercontent.com/media/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
set ONNX_MODEL="squeezenet.onnx"
SET ORT_PACKAGE=%1
SET ORT_DIR=%2
SET WORKSPACE=%3

cd %WORKSPACE%

echo The current directory is %CD%

7z.exe x %ORT_PACKAGE% -y
REM set ORT_LIB=%ORT_PACKAGE:~0,-4%\lib
REM echo %ORT_LIB%


cmake.exe -S . -B build\ -G "Visual Studio 16 2019"
cd build
powershell -Command "cp ..\%ORT_DIR%\lib\* ."
MSBuild.exe .\capi_test.sln /property:Configuration=Release
powershell -Command "cp ..\%ORT_DIR%\lib\* Release"
cd Release
powershell -Command "Invoke-WebRequest %ONNX_MODEL_URL% -Outfile %ONNX_MODEL%"
capi_test.exe
