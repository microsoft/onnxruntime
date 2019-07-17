:: Calls vswhere.exe from its known location

@echo off

call "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" %*
