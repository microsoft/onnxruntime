(This document is for ONNX Runtime developers to read, for seeking a common denominator of all Windows use cases that ONNX Runtime needs to support)

In terms of Windows running environment, roughly speaking ONNX Runtime supports three different build types:

1. For Windows Desktop apps (including console apps)
2. For [Universal Windows Platform (UWP) apps](https://docs.microsoft.com/en-us/windows/uwp/get-started/universal-application-platform-guide)
3. For Windows OneCore

## Desktop Build

The first one is the default and the one people are most familiar with. This build type has been in existence since the 1980s. The built binaries will have a dependency on kernel32.dll which provides the most important Win32 APIs such as CreateFileW. ONNX Runtime doesn't need to use any fancy Win32 API. So when you build it from source, the result binaries can support Windows versions down to Windows 7. It is "Windows 7" because in our top level CMakeLists.txt we added "WINVER=0x0601 \_WIN32\_WINNT=0x0601 NTDDI\_VERSION=0x06010000" macro definitions to all C/C++ source files. Here "0x0601" maps Windows 7.  You can find more details at https://docs.microsoft.com/en-us/cpp/porting/modifying-winver-and-win32-winnt . You can tweak the macros based on your needs as well.

You can always use the latest Windows SDK. For example, applications built with Windows 10 SDK can support Windows 7 as well. 

## UWP Build

UWP apps typically are for Windows Store. UWP apps can only use a subset of all Windows APIs. To accomplish this,  Windows SDK groups Windows APIs into multiple partitions. Each partition is a set of Windows APIs with a name. Partions may have overlaps, which means one Windows API can belong to mulitple paritions.  The following partitions are currently defined:

 * WINAPI_PARTITION_DESKTOP: usable for Desktop Win32 apps (but not store apps)
 * WINAPI_PARTITION_APP: usable for Windows Universal store apps
 * **WINAPI_PARTITION_PC_APP**: specific to Desktop-only store apps
 * WINAPI_PARTITION_PHONE_APP: specific to Phone-only store apps
 * WINAPI_PARTITION_SYSTEM: specific to System applications

Here we focus on Desktop-only store apps. **If a Windows API is not contained in the WINAPI_PARTITION_PC_APP parition, please do not use it here**.

As the above, you can always use the latest Windows SDK.

## OneCore Build

Windows OneCore is the common part of all Windows 10(or 11) Editions. Sometimes it is also called the core OS, or Windows Core OS(WCOS).  To build ONNX Runtime for Windows OneCore, please add "--enable_wcos" to the arguments of build.py(or build.bat). The binaries published in our Github release page and nuget.org are built in such a way. They take dependencies on [API sets](https://docs.microsoft.com/en-us/windows/win32/apiindex/windows-apisets) instead of kernel32.dll. Windows 10 supports two standard techniques to consume and interface with API sets: direct forwarding and reverse forwarding. Our OneCore build uses direct forwarding. Because this concept is only introduced since Windows 8, the prebuilt binaries we pulished can't run on Windows 7 and below. (However, our python package is built for Desktop so it can support Windows 7.)

If a Windows API is not in OneCore_apiset.lib of Windows 8, please do not use it here. For example, LocalFree. You need to get a Windows 8 SDK to see if the function is there.

# Other Special Considerations

## Virtual Trust Level (VTL)
Modern Windows uses a hypervisor assisted architecture to implement some of its core security features. It defined multiple [Virtual Trust Levels](https://docs.microsoft.com/en-us/virtualization/hyper-v-on-windows/tlfs/vsm). VTL0 is the least privileged level. Then VTL1, which ONNX Runtime must support. It limits what Windows APIs we can use.




