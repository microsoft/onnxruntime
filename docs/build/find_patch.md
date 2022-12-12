ONNX Runtime has a lot of third-party dependencies. Sometimes, we need to patch some of them to make the build pass. 
ONNX Runtime's cmake files use GNU's [patch tool](https://savannah.gnu.org/projects/patch/) to do the job. The tool 
is usually available on all \*nix/MacOS machines. However, Windows machines typically do not have it unless the machine has installed git(which is ported from Linux). If your machine has the patch tool but cmake cannot find it, please try the following steps:

# Add it to your PATH

Is it in your %PATH% environment? You can run "where patch" to figure it out. If you see: "INFO: Could not find files for the given pattern(s)", 
it means the tool is not in your PATH. You may add it there in a way like:


```batch
set PATH=C:\Program Files\git\usr\bin;%PATH%
```
(Do not put quote signs around the directory name)

Then run "where patch" again to verify the change, then run build.bat again in the same Windows.  Normally you don't need to do this, cmake should be able to
find the tool by searching the commonly known Git installation paths. If you want to know why it wasn't found at first, or if the above solution didn't work, please continue to read.

# Run cmake with diagnostic settings

You may run ONNX Runtime's build.bat(or build.py/build.sh) with `--cmake_extra_defines CMAKE_FIND_DEBUG_MODE=ON`, then it will tell you which directories were used
to find the patch tool. You may also create a minimal cmake project as below and try the same thing.

```cmake
cmake_minimum_required(VERSION 3.24)

project(onnxruntime C CXX ASM)
find_package(Patch)
```

Save the text above to a file with name "CMakeLists.txt", and put the file in to a newly created empty directory, then run `cmake .  -DCMAKE_FIND_DEBUG_MODE=ON >1.log 2>&1` in the directory.
If this cmake project can find `patch` but ONNX Runtime's didn't, you can compare the two log files to find the difference. 

# Check if you were using a toolchain file

Please check onnxruntime's build.bat(or build.sh/build.py)'s output to see if it contains words like:  '-DCMAKE_TOOLCHAIN_FILE=something.cmake`. Toolchain files are used for cross-compiling,
which means the target environment(where the final bits will run) is different than the host environment(where your compiler runs), therefore cmake would avoid to search 
directories like %PROGRAMFILES%, %ProgramData%. Then you may need to do extra things to help cmake find the tools(or libraries/header files) we need.

# Set the Patch_EXECUTABLE cmake variable

If cmake cannot find the patch tool but you know where it is, you may give cmake a hint by adding `--cmake_extra_defines Patch_EXECUTABLE=yourexepath` to your build command.
