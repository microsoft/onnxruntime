It's often there are multiple ways of doing the same thing. That's why we have this guideline.  This is not about which is correct/wrong. It is for aligning us to the same direction.

# Scope the impact to minimal
If you want to change some setting, please try to scope down the impact to be local. 

- Prefer target\_include\_directories to include\_directories
- Prefer target\_compile\_definitions to add\_definitions
- Prefer target\_compile\_options to add\_compile\_options
- Don't touch the global flags like CMAKE\_CXX\_FLAGS

For example, to add a macro definition to one VC project, you should use target\_compile\_definitions, not the add\_definitions.

# Setting CMAKE_C_FLAGS/CMAKE_CXX_FLAGS/CMAKE_CXX_FLAGS_DEBUG/...
CMake supports multiple programming languges. For example, C, C++, CUDA, assembly.      
CMAKE_C_FLAGS/CMAKE_CXX_FLAGS are the default compile flags for every C/C++ source file. For example, in a Linux build `CMAKE_CXX_FLAGS` is default empty.     
If you want to **override** the default value, you may pass `-DCMAKE_CXX_FLAGS=soemthing` to doing cmake config. For example:
```bash
cmake .. -DCMAKE_CXX_FLAGS="-fno-exception"
```
Our build.py has a lot of such logic.

 CMAKE_CXX_FLAGS_DEBUG/CMAKE_CXX_FLAGS_RELEASE/... flags per config flags that are appended after CMAKE_CXX_FLAGS. For example, in a Linux build `CMAKE_CXX_FLAGS_DEBUG` is default to `-g`. If you want to override the flags, please be careful that you must keep `-DNDEBUG` in non-Debug builds.

If you want to append values to the flags(instead of overriding the default), you should set `CFLAGS`/`CXXFLAGS` environment variables instead. Like
```bash
CXXFLAGS="-frtti"  cmake ..
```
If `CXXFLAGS` env var and `CMAKE_CXX_FLAGS` are both used, `CMAKE_CXX_FLAGS` takes higher precedence and the other one gets ignored. 

Our [adjust_global_compile_flags.cmake](/cmake/adjust_global_compile_flags.cmake) also has logics of appending C/C++ flags. It's another way of doing that. 

The above settings affects all ONNX Runtime C/C++ code and third-party libraries(like protobuf) when vcpkg is not used.
When vcpkg is enabled, they do not affect third-party code. The compile flags for the dependent packages are controlled by the triplets files in [/cmake/vcpkg-triplets](/cmake/vcpkg-triplets). Since we support both vcpkg and non-vcpkg build, it's crucial to make them consistent.  For example, in [adjust_global_compile_flags.cmake](/cmake/adjust_global_compile_flags.cmake)  we have:
```cmake
  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    string(APPEND CMAKE_C_FLAGS " -msimd128")
    string(APPEND CMAKE_CXX_FLAGS " -msimd128")
  endif()
```
It means in `vcpkg-triplets` folder we need to have two different triplets that either with the "-msimd128" flag or not. 

# Static library order matters
First, you should know, when you link static libraries to an executable(or shared library) target, the order matters a lot.
Let's say, if A and B are static libraries, C is an exe.      
- A depends B.
- C depends A and B.    

Then you should write
```
target_link_libraries(C PRIVATE A B)
```
Not
```
target_link_libraries(C PRIVATE B A)  #Wrong!
```

On Windows, the order of static libraries does matter if a symbol is defined in more than one library.       
On Linux, it matters when one static library references another.

So, in general, please always put them in right order (according to their dependency relationship).

Example:

CMakeLists.txt:
```cmake
project(test1 C CXX)

add_library(test1 STATIC test1.c)
add_library(test2 STATIC test2.c)
add_executable(test3 main.cpp)
target_link_libraries(test3 PRIVATE test1 test2)
```

test1.c:
```c
#include <stdio.h>

void foo(){
  printf("hello foo\n");
}
```

test2.c:
```c
#include <stdio.h>

extern void foo();

void foo2(){
  foo();
  printf("hello foo2\n");
}
```

main.cpp
```c++
#include <iostream>

extern "C" {
  extern void foo2();
}

int main(){
  foo2();
  return 0;
}
```
Then when you build the project, it will report
```
/usr/bin/ld: libtest2.a(test2.c.o): in function `foo2':
test2.c:(.text+0xa): undefined reference to `foo'
collect2: error: ld returned 1 exit status
```
But if you change
```cmake
target_link_libraries(test3 PRIVATE test1 test2)
```
to 
```cmake
target_link_libraries(test3 PRIVATE test2 test1)
```
It will be fine.

Or if you change the two libraries to shared libs, build will also pass.

# Don't call target\_link\_libraries on static libraries
You could do it, but please don't.

As we said before, library order matters. If you explicitly list all the libs in one line, and if some libs were in wrong position, it's easy to fix. 

However, if any static lib was built with target\_link\_libraries,
- First you should know ,there is no link step for a static lib
- Second, once you hit the ordering problem, it would be harder to fix. Because many of the deps were implicit, and their position would be out of our control.

# Every linux program(and shared lib) should link to libpthread and libatomic
In Linux world, there are two set of pthread symbols. A fake one in the standard c library, and a real one in pthread.so. If the real one is not loaded while the process was starting up, then the process shouldn't use multiple threading because the core part was missing.

So, We append "Threads::Threads" to the lib list of every shared lib(*.so,*.dll) and exe target. It's easy to get missed. If it happened, the behavior is undefined.

Another related thing is: if std::atomic was in use, please also add the atomic lib there. Because some uses of std::atomic require linking to libatomic. see [https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_concurrency.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_concurrency.html)

NOTE: However, in rare cases, even you told linker to link the program with pthread, sometimes it doesn't listen to you. It may ignore your order, cause issues. see [https://github.com/protocolbuffers/protobuf/issues/5923](https://github.com/protocolbuffers/protobuf/issues/5923). 

# The "-pthread" flag
"-pthread" is a flag for GCC/Clang compilers and it work in both compiling and linking stage. Howevery, usually you do not need to manually specify the flag. Historically(in the previous century) you need to apply this flag when compiling multithreading C/C++ code. But it is no longer true. Now for all the platforms we support it is only needed for WebAssembly build. It is useless for Linux build that is based on Glibc. 

We still need to add "Threads::Threads" to link libraries, which effectively will add "-pthread" to the link command.

# CUDA projects should use the new cmake CUDA approach
There are two ways of enabling CUDA in cmake.
1. (new): enable\_language(CUDA)
2. (old): find\_package(CUDA)

Use the first one, because the second one is deprecated. Don't use ["find\_package(CUDA)"](https://cmake.org/cmake/help/latest/module/FindCUDA.html). It also means, don't use the vars like:

- CUDA\_NVCC\_FLAGS
- CUDA\_INCLUDE\_DIRS
- CUDA\_LIBRARIES

So, be careful on this when you copy code from another project to ours, the changes may not work.

# Basics of cross-compiling
Host System: The system where compiling happens
Target System: The system where built programs and libraries will run.

Here system means the combination of

- CPU Arch: x86_32, x86_64, armv6, armv7, arvm7l, aarch64, …
- OS: bare-metal, linux, Windows
- Libc: gnu libc/ulibc/musl/…
- ABI: ARM has mutilple ABIs like eabi, eabihf…

When "host system" != "target system" (any different in the four dimensions), we call it cross-compiling. For example, when you build a Windows EXE on Linux, or build an ARM program on an x86_64 CPU, you are doing cross-compiling. Then special handling is needed.

For example, while you build the whole code base for the target system, you must build protoc for the host system. And because protoc also depends on libprotobuf, you will need to build libprotobuf twice: for the host and for the target.

Here we focus on what you should know when authoring cmake files.

# How to determine the host CPU architecture: on which cmake is running
CMAKE_HOST_SYSTEM_PROCESSOR is the one you should use. 

What are the valid values:
- macOS: it can be x86_64 or arm64. (maybe it could also be arm64e but cmake forgot to document that)
- Linux: i686, x86_64, aarch64, armv7l, ... The possible values for `uname -m` command. They slightly differ from what you can get from GCC. This sometimes confuses people: `cmake` and `uname` sit in one boat, GCC is in another boat but GCC is closer to your C/C++ source code.
- Windows: AMD64, ...
- Android/iOS/...: we don't care. We don't use them as a development environment.

# How to determine what CPU architecture(or architectures) you are building for

Linux: Use `CMAKE_SYSTEM_PROCESSOR`. When not cross-compiling, you can read this value but please don't set it. This variable has the same value as `CMAKE_HOST_SYSTEM_PROCESSOR`. When cross-compiling, a CMAKE_TOOLCHAIN_FILE should set the `CMAKE_SYSTEM_PROCESSOR` variable to match target architecture. But, I used to forgot it and it didn't cause any problem until pytorch cpuinfo was added into onnx runtime as a dependency. For simplicity, let's assume people won't miss it.

macOS: Don't use `CMAKE_SYSTEM_PROCESSOR`. First, please check if target property `OSX_ARCHITECTURES` or `CMAKE_OSX_ARCHITECTURES` was set. Please note it is a list which could contain multiple values, like "arm64;x86_64". Otherwise please use`CMAKE_HOST_SYSTEM_PROCESSOR`. 

In all cases, possible values are the same as what we talked above(for `CMAKE_HOST_SYSTEM_PROCESSOR`).

If you have platform-specific code, you can wrap it with conditional compilation macros("#ifdef"). But if you have a file that is solely for one specific cpu architecture without the macros, you need to put it in a separated lib, like libfoo_x86, libfoo_arm64, ... 








