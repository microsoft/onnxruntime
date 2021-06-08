---
title: Build for Android/iOS
parent: Build ORT
nav_order: 5
grand_parent: How to
---

# Build ONNX Runtime for Android and iOS
{: .no_toc }

Below are general build instructions for Android and iOS. For instructions on fully deploying ONNX Runtime on mobile platforms (includes overall smaller package size and other configurations), see [How to: Deploy on mobile](../mobile).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Android

### Prerequisites

The SDK and NDK packages can be installed via Android Studio or the sdkmanager command line tool.

Android Studio is more convenient but a larger installation.
The command line tools are smaller and usage can be scripted, but are  a little more complicated to setup. They also require a Java runtime environment to be available.

Resources:
- [API levels](https://developer.android.com/guide/topics/manifest/uses-sdk-element.html)
- [Android ABIs](https://developer.android.com/ndk/guides/abis)
- [System Images](https://developer.android.com/topic/generic-system-image)

#### Android Studio

1. [Install](https://developer.android.com/studio) Android Studio

2. Install any additional SDK Platforms if necessary
  * File->Settings->Appearance & Behavior->System Settings->Android SDK to see what is currently installed
  * Note that the SDK path you need to use as --android_sdk_path when building ORT is also on this configuration page
  * Most likely you don't require additional SDK Platform packages as the latest platform can target earlier API levels.

3. Install an NDK version
  * File->Settings->Appearance & Behavior->System Settings->Android SDK
  * 'SDK Tools' tab
    * Select 'Show package details' checkbox at the bottom to see specific versions. By default the latest will be installed which should be fine.
  * The NDK path will be the 'ndk/{version}' subdirectory of the SDK path shown
    * e.g. if 21.1.6352462 is installed it will be {SDK path}/ndk/21.1.6352462

#### sdkmanager from command line tools

* If necessary install the Java Runtime Environment and set the JAVA_HOME environment variable to point to it
  * https://www.java.com/en/download/
  * Windows note: You MUST install the 64-bit version (https://www.java.com/en/download/manual.jsp) otherwise sdkmanager will only list x86 packages and the latest NDK is x64 only.
* For sdkmanager to work it needs a certain directory structure. First create the top level directory for the Android infrastructure.
  * in our example we'll call that `.../Android/`
* Download the command line tools from the 'Command line tools only' section towards the bottom of https://developer.android.com/studio
* Create a directory called 'cmdline-tools' under your top level directory
  * giving `.../Android/cmdline-tools`
* Extract the 'tools' directory from the command line tools zip file into this directory
  * giving `.../Android/cmdline-tools/tools`
  * Windows note: preferably extract using 7-zip. If using the built in Windows zip extract tool you will need to fix the directory structure by moving the jar files from `tools\lib\_` up to `tools\lib`
    * See https://stackoverflow.com/questions/27364963/could-not-find-or-load-main-class-com-android-sdkmanager-main
* You should now be able to run Android/cmdline-tools/bin/sdkmanager[.bat] successfully
  * if you see an error about it being unable to save settings and the sdkmanager help text,
      your directory structure is incorrect.
  * see the final steps in this answer to double check: https://stackoverflow.com/a/61176718

* Run `.../Android/cmdline-tools/bin/sdkmanager --list` to see the packages available

* Install the SDK Platform
  * Generally installing the latest is fine. You pick an API level when compiling the code and the latest platform will support many recent API levels e.g.

    ```
    sdkmanager --install "platforms;android-29"
    ```

  * This will install into the 'platforms' directory of our top level directory, the `Android` directory in our example
  * The SDK path to use as `--android_sdk_path` when building is this top level directory

* Install the NDK
  * Find the available NDK versions by running `sdkmanager --list`
  * Install
    * you can install a specific version or the latest (called 'ndk-bundle') e.g. `sdkmanager --install "ndk;21.1.6352462"`
    * NDK path in our example with this install would be `.../Android/ndk/21.1.6352462`
    * NOTE: If you install the ndk-bundle package the path will be `.../Android/ndk-bundle` as there's no version number

### Android Build Instructions

#### Cross compiling on Windows

The [Ninja](https://ninja-build.org/) generator needs to be used to build on Windows as the Visual Studio generator doesn't support Android.

```powershell
./build.bat --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)> --cmake_generator Ninja
```

e.g. using the paths from our example

```
./build.bat --android --android_sdk_path .../Android --android_ndk_path .../Android/ndk/21.1.6352462 --android_abi arm64-v8a --android_api 27 --cmake_generator Ninja
```

#### Cross compiling on Linux and macOS

```bash
./build.sh --android --android_sdk_path <android sdk path> --android_ndk_path <android ndk path> --android_abi <android abi, e.g., arm64-v8a (default) or armeabi-v7a> --android_api <android api level, e.g., 27 (default)>
```

#### Build Android Archive (AAR)

Android Archive (AAR) files, which can be imported directly in Android Studio, will be generated in your_build_dir/java/build/android/outputs/aar, by using the above building commands with `--build_java`

To build on Windows with `--build_java` enabled you must also:

* set JAVA_HOME to the path to your JDK install
  * this could be the JDK from Android Studio, or a [standalone JDK install](https://www.oracle.com/java/technologies/javase-downloads.html)
  * e.g. Powershell: `$env:JAVA_HOME="C:\Program Files\Java\jdk-15"` CMD: `set JAVA_HOME=C:\Program Files\Java\jdk-15`
* install [Gradle](https://gradle.org/install/) and add the directory to the PATH
  * e.g. Powershell: `$env:PATH="$env:PATH;C:\Gradle\gradle-6.6.1\bin"` CMD: `set PATH=%PATH%;C:\Gradle\gradle-6.6.1\bin`
* run the build from an admin window
  * the Java build needs permissions to create a symlink, which requires an admin window

### Android NNAPI Execution Provider

If you want to use NNAPI Execution Provider on Android, see [NNAPI Execution Provider](../../reference/execution-providers/NNAPI-ExecutionProvider).

#### Build Instructions

Android NNAPI Execution Provider can be built using building commands in [Android Build instructions](#android-build-instructions) with `--use_nnapi`

---

## iOS

### Prerequisites

* A Mac computer with latest macOS
* Xcode, https://developer.apple.com/xcode/
* CMake, https://cmake.org/download/
* Python 3, https://www.python.org/downloads/mac-osx/

### General Info

* iOS Platforms

  The following two platforms are supported
  * iOS device (iPhone, iPad) with arm64 architecture
  * iOS simulator with x86_64 architecture

  The following platforms are *not* supported
  * armv7
  * armv7s
  * i386 architectures
  * tvOS
  * watchOS platforms are not currently supported.

* apple_deploy_target

  Specify the minimum version of the target platform (iOS) on which the target binaries are to be deployed.

* Code Signing

  If the code signing development team ID or code signing identity is specified, and has a valid code signing certificate, Xcode will code sign the onnxruntime library in the building process. Otherwise, the onnxruntime will be built without code signing. It may be required or desired to code sign the library for iOS devices. For more information, see [Code Signing](https://developer.apple.com/support/code-signing/).

* Bitcode

  Bitcode is an Apple technology that enables you to recompile your app to reduce its size. It is by default enabled for building onnxruntime. Bitcode can be disabled by using the building commands in [iOS Build instructions](#build-instructions-1) with `--apple_disable_bitcode`. For more information about bitcode, please see [Doing Basic Optimization to Reduce Your App’s Size](https://developer.apple.com/documentation/xcode/doing-basic-optimization-to-reduce-your-app-s-size).

### Build Instructions

Run one of the following build scripts from the ONNX Runtime repository root:

#### Cross build for iOS simulator

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphonesimulator --osx_arch x86_64 --apple_deploy_target <minimal iOS version>
```

#### Cross build for iOS device

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version>
```

#### Cross build for iOS device and code sign the library using development team ID

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version> \
           --xcode_code_signing_team_id <Your Apple developmemt team ID>
```

#### Cross build for iOS device and code sign the library using code sign identity

```bash
./build.sh --config <Release|Debug|RelWithDebInfo|MinSizeRel> --use_xcode \
           --ios --ios_sysroot iphoneos --osx_arch arm64 --apple_deploy_target <minimal iOS version> \
           --xcode_code_signing_identity <Your preferred code sign identity>
```

### CoreML Execution Provider

If you want to use CoreML Execution Provider on iOS or macOS, see [CoreML Execution Provider](../../reference/execution-providers/CoreML-ExecutionProvider).

#### Build Instructions

CoreML Execution Provider can be built using building commands in [iOS Build instructions](#build-instructions-1) with `--use_coreml`
