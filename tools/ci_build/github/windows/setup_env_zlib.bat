@echo off
set search_path=%1\vcpkg\src\vcpkg\installed
for /r %search_path% %%a in (*.dll) do (
  if "%%~nxa"=="zlib1.dll" (
    echo Found zlib1.dll under: %%~dpa
    set zlib=%%~dpa
    goto FOUND
  )
)
echo No zlib1.dll found, exit
goto EOF
:FOUND
set PATH=%PATH%;%zlib%
:EOF