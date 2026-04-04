@echo off
setlocal

:: This script requires an environment with MSVC toolchain (cl.exe).
:: Open x64 Native Tools Command Prompt for VS to run this.

echo ==============================================
echo 1. Creating dummy dependency DLL...
echo ==============================================
echo __declspec(dllexport) void dep_func() {} > missing_dep.cpp
cl /LD /nologo missing_dep.cpp /link /OUT:missing_dep.dll
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ==============================================
echo 2. Creating dummy main DLL that depends on missing_dep.dll...
echo ==============================================
echo void __declspec(dllimport) dep_func(); extern "C" __declspec(dllexport) void* RegisterCustomOps(void* options, const void* api) { dep_func(); return nullptr; } > main.cpp
cl /LD /nologo main.cpp /link /OUT:test_main.dll missing_dep.lib
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo ==============================================
echo 3. Deleting missing_dep.dll to trigger dependent load error...
echo ==============================================
del missing_dep.dll
if exist missing_dep.dll (
    echo FAILED: Could not delete missing_dep.dll
    exit /b 1
)

echo ==============================================
echo 4. Running the Python test script...
echo ==============================================
python test_dll_load.py
set PY_ERROR=%ERRORLEVEL%

echo ==============================================
echo 5. Cleaning up dummy files...
echo ==============================================
del missing_dep.cpp main.cpp
del missing_dep.lib missing_dep.exp missing_dep.obj
del test_main.dll main.lib main.exp main.obj test_main.lib test_main.exp test_main.obj 2>nul

exit /b %PY_ERROR%
endlocal
