It's often there are multiple ways of doing the same thing. That's why we have this guideline.  This is not about which is correct/wrong. It is for aligning us to the same direction.

# Scope the impact to minimal
If you want to change some setting, please try to scope down the impact to be local. 

- Prefer target\_include\_directories to include\_directories
- Prefer target\_compile\_definitions to add\_definitions
- Prefer target\_compile\_options to add\_compile\_options
- Don't touch the global flags like CMAKE\_CXX\_FLAGS

For example, to add a macro definition to one VC project, you should use target\_compile\_definitions, not the add\_definitions.


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

# Don't use the "-pthread" flag directly. 
Because:
1. It doesn't work with nvcc(the CUDA compiler)
2. Not portable. 

Don't bother to add this flag to your compile time flags. On Linux, it's useless. On some very old unix-like system, it may be helpful, but we only support Ubuntu 16.04.

Use "Threads::Threads" for linking. Use nothing for compiling.

# CUDA projects should use the new cmake CUDA approach
There are two ways of enabling CUDA in cmake.
1. (new): enable\_language(CUDA)
2. (old): find\_package(CUDA)

Use the first one, because the second one is deprecated. Don't use ["find\_package(CUDA)"](https://cmake.org/cmake/help/latest/module/FindCUDA.html). It also means, don't use the vars like:

- CUDA\_NVCC\_FLAGS
- CUDA\_INCLUDE\_DIRS
- CUDA\_LIBRARIES

So, be careful on this when you copy code from another project to ours, the changes may not work.










