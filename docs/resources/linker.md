
An executable file links to a library in one of three ways:

1. Static linking, where the linking is done at build time.

2. Implicit dynamic linking, where the operating system loads the dynamic library at the same time 
as the executable that uses it.

3. Explicit dynamic linking, where the operating system loads the dynamic library on demand at 
runtime. From the definition, DLL delay loading is actually one kind of "Explicit dynamic linking". 

## Static linking ##
Static linking is simple and mature, it delivers the best run time performance, and most of the 
linking issues can be caught ahead at build time. There are a few things you probably should know:

1. There is no link stage for a static library. You can't ask a static library linking to another
 library. Even cmake allows you to do that, I feel it can cause more confusion than useful.

2. Sometimes the order of the libraries is important. On Linux a library on the left can depend 
on the libraries on its right, but not vice versa. On Windows the rule is relaxed that as long as
no symbol is defined in more than one library, it doesn't matter. 

3. On Windows, source file has the highest precedence in symbol searching, and it is the only 
reliable way to override a symbol(e.g. malloc/free) which is already provided at another place. In
other words, some code are not suitable to be compiled to a static library.


##When are global objects constructed and destructed by Visual C++? ##

| When does it run?	  | Constructor  | Destructor  |
|---|---|---|
|magic statics in EXE	   |  the first use	  | Just before before any atexit calls or static object destructors  |
|magic statics in DLL	   |  the first use	  | (TODO)|
|Global object in EXE	   |  C runtime startup code	  | When processing atexit calls   |
|Global object in DLL	   | C runtime DLL_PROCESS_ATTACH prior to Dll­Main	  |  C runtime DLL_PROCESS_DETACH after Dll­Main returns
  |

TODO: examine the variables without a non-trivial constructor.

## Delay Loading ##
[Delay Loading](https://docs.microsoft.com/en-us/cpp/build/reference/linker-support-for-delay-loaded-dlls?view=msvc-160) 
is only available on Windows.

Q: If A.dll depends on B.dll, what is required to delay load B.dll?

A: [If the initialization and deinitialization of A.dll totally doesn't depend on B.dll.](https://devblogs.microsoft.com/oldnewthing/20190718-00/?p=102719).
Also, please read [Constraints of Delay Loading DLLs](https://docs.microsoft.com/en-us/cpp/build/reference/constraints-of-delay-loading-dlls?view=msvc-160).

Q: Why can't we delay load the CUDA DLLs?

A: Because our CUDA EP has some thread local variables that may call CUDA functions in their 
destructors. And we don't have a way to ensure all such variables has been cleaned up when the 
corresponding session is closed. See https://github.com/microsoft/onnxruntime/pull/3147


##Switching between static linking and dynamic linking ##

For some reasons, you can't freely change it.

### Init order of global variables ###
The init order of global variables may get messed up when you convert a library from dynamic 
linking to static linking. For example, if you have an application that statically links to google 
protobuf, and if the application has a global variable with a non-trivial constructor which uses
 some symbols from protobuf, then the behavior result is undefined because protobuf also has many global
variables and protobuf may get initialized too late. See [What’s the “static initialization order ‘fiasco’ (problem)”?](https://isocpp.org/wiki/faq/ctors#static-init-order).
And, unfortunately such errors are hard to detect.     

###Unwanted sharing ###
When you change a static library to dynamic, the change may extend
some variables' visibility and lifetime. For example, [ONNX](https://github.com/onnx/onnx) 
is an open standard and open source library for machine learning interoperability built upon libprotobuf. 
ONNX was designed to be sharable and reusable among machine learning frameworks. And libprotobuf
has a functionality which allows you create a message based on its
name(we call reflection), so protobuf requires you have a centralized way to managed all the protobuf
 definitions (*.proto files) to avoid name conflicts. But commonly it's not a real conflict, it's just 
 two pieces of code want to use the same protobuf definition. Like if you have two ML frameworks in
python package form that both dynamically link to libprotobuf, then they must dynamically link to ONNX too, 
so that the protobuf messages defined in ONNX will only have one copy per process and will only be 
registered once to libprobuf's central database. Also none of the frameworks can extend or reuse 
the message definition in a different way. If a framework wants to define a protobuf message that 
 contains an ONNX TensorProto message inside, it can not directly include ONNX's existing *.proto file. 


## PE and ELF
[Portable Executable (PE)](https://docs.microsoft.com/en-us/windows/win32/debug/pe-format) and 
[Executable and Linking Format (ELF)](https://man7.org/linux/man-pages/man5/elf.5.html) are the most 
popular file format for executables and shared libraries under the Windows and Linux family of 
operating systems. Here I will not go into the details but I will tell you some major differences 
between these two:   

### Relocatable Image vs Position Independent Code
When you build a DLL, historically you'll need to set a [preferred base address](https://docs.microsoft.com/en-us/cpp/build/reference/base-base-address?view=msvc-160) 
for it. If you set the address explicitly, by default it is 0x10000000 for 32-bit images or 0x180000000 
for 64-bit images. Then at build time, linker will use this address to calculate all the offsets, and 
at runtime if DLL loader find the DLL can be well fit into the address, then that's perfect, the DLL 
can be paged in directly from disk to main memory and the in memory image can be shared between 
multiple processes to reduce the overall memory footprint. Otherwise the address of every absolute 
jump in the code for the executable need be fixed up and relocated to a different address. So we say
DLLs are "Relocatable".
 
Nowadays, most DLLs are built with [address 
space layout randomization (ASLR)](https://docs.microsoft.com/en-us/cpp/build/reference/dynamicbase-use-address-space-layout-randomization?view=msvc-160) 
and Windows operating system will randomly pick up a base address for you. In this case, relocation 
always happens.

Linux has a different approach: the compiler can generate all the code in a position independent way
so that there is no need to fix up(relocate) the code at runtime. However, it comes with a performance
penalty too. So by default it is not enabled and you need to manually add the following code to your
cmake file to turn it on:

```
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
```

So, if you want to build a shared library(libfoo.so) which static link to another library(libbar.a), 
then typically you need to manually rebuild the second one(libbar.a) from source with the 
POSITION INDEPENDENT CODE option is ON.


In summary:

|   | Relocatable Image  | Position Independent Code  |
|---|---|---|
|Windows   |  Yes  | No   |
|Linux x86   | Yes, but not popular  |  Yes  |
|Linux x64   | No  |  Yes |


## The use of RPATH:
(In this section we only talk implicit dynamic linking.)

Dynamic Section is an optional section in ELF files for holding relevant dynamic linking information.
You can use the "readelf" tool to view it.
```
$ readelf -d onnxruntime_perf_test 

Dynamic section at offset 0x26bd00 contains 33 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libonnxruntime.so.1.5.2]
 0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
 0x0000000000000001 (NEEDED)             Shared library: [librt.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000f (RPATH)              Library rpath: [$ORIGIN:/home/chasun/src/gcc]
 0x000000000000000c (INIT)               0x408000
 0x000000000000000d (FINI)               0x5a15b4
 0x0000000000000019 (INIT_ARRAY)         0x665590
 0x000000000000001b (INIT_ARRAYSZ)       296 (bytes)
 0x000000000000001a (FINI_ARRAY)         0x6656b8
 0x000000000000001c (FINI_ARRAYSZ)       8 (bytes)
 0x000000006ffffef5 (GNU_HASH)           0x400340
 0x0000000000000005 (STRTAB)             0x401ea0
 0x0000000000000006 (SYMTAB)             0x4003a0
 0x000000000000000a (STRSZ)              10429 (bytes)
 0x000000000000000b (SYMENT)             24 (bytes)
 0x0000000000000015 (DEBUG)              0x0
 0x0000000000000003 (PLTGOT)             0x66d000
 0x0000000000000002 (PLTRELSZ)           6144 (bytes)
 0x0000000000000014 (PLTREL)             RELA
 0x0000000000000017 (JMPREL)             0x4066d0
 0x0000000000000007 (RELA)               0x404ba0
 0x0000000000000008 (RELASZ)             6960 (bytes)
 0x0000000000000009 (RELAENT)            24 (bytes)
 0x000000006ffffffe (VERNEED)            0x4049a0
 0x000000006fffffff (VERNEEDNUM)         7
 0x000000006ffffff0 (VERSYM)             0x40475e
 0x0000000000000000 (NULL)               0x0
```

RPATH is an array of hardcoded paths in dynamic section to specify directories for searching for the 
`NEEDED` dependencies. In this example, the array has two entries:"$ORIGIN" and "/home/chasun/src/gcc".
But you don't want to see the second one in any of our published prebuilt binaries, because it is an 
hardcoded absolute path that should only exist on my dev machine. Also, indeed the first entry is 
not needed too.

RPATH is old and should be deprecated, but it has the highest precedence compared to the 
other similar mechanisms(like LD_LIBRARY_PATH). When used properly, it is the most reliable
and actually very good.

So, how should we use it in our project? Just use the default cmake settings.

1. At build time cmake will link the executables and shared libraries with full RPATH to all used 
libraries in the build tree. So all the binaries work perfectly on the build machine, they would never
pick up a wrong dependency.

2. When installing, cmake will clear the RPATH of these targets so they are installed with an empty 
RPATH. So we assume there is an "installing" step before we transfer the compiled binaries to another
machine.

3. When building a Linux python package, we skip the second step and let the [auditwheel](https://github.com/pypa/auditwheel) 
tool to do the same thing. 

4. At run time, end users can use the environment variable LD_LIBRARY_PATH to adjust the directories 
the loader searches. 
