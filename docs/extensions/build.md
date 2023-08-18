---
title: Build
description: Instructions for building and developing ORT Extensions.
parent: Extensions
nav_order: 3
---
# Build from Source

This project supports Python and can be built from source easily, or a simple cmake build without Python dependency.
## Python package
The package contains all custom operators and some Python scripts to manipulate the ONNX models.
- Install Visual Studio with C++ development tools on Windows, or gcc(>8.0) for Linux or xcode for macOS, and cmake on the unix-like platform. (**hints**: in Windows platform, if cmake bundled in Visual Studio was used, please specify the set _VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\<VERSION_YEAR>\<Edition>\Common7\Tools\VsDevCmd.bat_)
- If running on Windows, ensure that long file names are enabled, both for the [operating system](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd) and for git: `git config --system core.longpaths true`
- Prepare Python env and install the pip packages in the requirements.txt.
- `pip install .` to build and install the package.<br/> OR `pip install -e .` to install the package in the development mode, which is more friendly for the developer since the Python code change will take effect without having to copy the files to a different location in the disk.(**hints**: debug=1 in setup.cfg wil make C++ code be debuggable in a Python process.)

Test:
- 'pip install -r requirements-dev.txt' to install pip packages for development.
- run `pytest test` in the project root directory.

For a complete list of verified build configurations see [here](./build.md#dependencies)

## Java package

Run `bash ./build.sh -DOCOS_BUILD_JAVA=ON` to build jar package in out/<OS>/Release folder

## Android package
- pre-requisites: [Android Studio](https://developer.android.com/studio)

Use `./tools/android/build_aar.py` to build an Android AAR package.

## iOS package
Use `./tools/ios/build_xcframework.py` to build an iOS xcframework package.

## Web-Assembly
ONNXRuntime-Extensions will be built as a static library and linked with ONNXRuntime due to the lack of a good dynamic linking mechanism in WASM. Here are two additional arguments [–-use_extensions and --extensions_overridden_path](https://github.com/microsoft/onnxruntime/blob/860ba8820b72d13a61f0d08b915cd433b738ffdc/tools/ci_build/build.py#L416) on building onnxruntime to include ONNXRuntime-Extensions footprint in the ONNXRuntime package.

## The C++ shared library
for any other cases, please run `build.bat` or `bash ./build.sh` to build the library. By default, the DLL or the library will be generated in the directory `out/<OS>/<FLAVOR>`. There is a unit test to help verify the build.


**VC Runtime static linkage**
If you want to build the binary with VC Runtime static linkage, please add a parameter _-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"_ on running build.bat

## Copyright guidance
check this link https://docs.opensource.microsoft.com/releasing/general-guidance/copyright-headers/ for source file copyright header.



# Build ONNX Runtime with onnxruntime-extensions for Java package

*The following step are demonstrated for Windows Platform only, the others like Linux and MacOS can be done similarly.*

> Android build was supported as well; check [here](https://onnxruntime.ai/docs/build/android.html#cross-compiling-on-windows) for arguments to build AAR package.

## Tools required
1. install visual studio 2022 (with cmake, git, desktop C++)
2. install miniconda to have Python support (for onnxruntime build)
3. OpenJDK: https://docs.microsoft.com/en-us/java/openjdk/download
		(OpenJDK 11.0.15 LTS)
4. Gradle: https://gradle.org/releases/
		(v6.9.2)

## Commands
Launch **Developer PowerShell for VS 2022** in Windows Tereminal
```
	. $home\miniconda3\shell\condabin\conda-hook.ps1
	conda activate base

	$env:JAVA_HOME="C:\Program Files\Microsoft\jdk-11.0.15.10-hotspot"
	# clone ONNXRuntime
	git clone -b rel-1.12.0 https://github.com/microsoft/onnxruntime.git onnxruntime

	# clone onnxruntime-extensions
	git clone https://github.com/microsoft/onnxruntime-extensions.git onnxruntime_extensions

	# build JAR package in this folder
	mkdir ortall.build
	cd ortall.build
	python ..\onnxruntime\tools\ci_build\build.py --config Release --cmake_generator "Visual Studio 17 2022" --build_java --build_dir .  --use_extensions --extensions_overridden_path "..\onnxruntime-extensions"
```


## Dependencies

The matrix below lists the versions of individual dependencies of onnxruntime-extensions. These are the configurations that are routinely and extensively verified by our CI.

Python | 3.8 | 3.9 | 3.10 | 3.11 |
---|---|---|---|---
Onnxruntime |1.12.1 (Aug 4, 2022) |1.13.1(Oct 24, 2022)  |1.14.1 (Mar 2, 2023) |1.15.0 (May 24, 2023) |
