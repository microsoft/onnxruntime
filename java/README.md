# ONNX Runtime Java API

This directory contains the Java language binding for the ONNX runtime.
Java Native Interface (JNI) is used to allow for seamless calls to ONNX runtime from Java.

## Usage

TBD: maven distribution

The minimum supported Java Runtime is version 8.

An example implementation is located in `src/test/java/sample/ScoreMNIST.java`

This project can be built manually using the instructions below.

### Building

Use the main project's [build instructions](../BUILD.md) with the `--build_java` option.

#### Requirements

JDK version 8 or later is required.
The [Gradle](https://gradle.org/) build system is required and used here to manage the Java project's dependency management, compilation, testing, and assembly.
You may use your system Gradle installation installed on your PATH.
Version 6 or newer is recommended.
Optionally, you may use your own Gradle [wrapper](https://docs.gradle.org/current/userguide/gradle_wrapper.html) which will be locked to a version specified in the `build.gradle` configuration.
This can be done once by using system Gradle installation to invoke the wrapper task in the java project's directory: `cd REPO_ROOT/java && gradle wrapper`
Any installed wrapper is gitignored.

#### Build Output

The build will generate output in `$REPO_ROOT/build/$OS/$CONFIGURATION/java/build`:

* `docs/javadoc/` - HTML javadoc
* `reports/` - detailed test results and other reports
* `libs/onnxruntime.jar` - JAR with classes, depends on `onnxruntime-jni.jar` and `onnxruntime-lib.jar `
* `libs/onnxruntime-jni.jar`- JAR with JNI shared library
* `libs/onnxruntime-lib.jar` - JAR with onnxruntime shared library
* `libs/onnxruntime-all.jar` - basically the 3 preceding jars all combined: JAR with classes, JNI shared library, and onnxruntime shared library

The reason the shared libraries are split out like that is that users can mix and match to suit their use case:

* To support a single OS/Architecture without any dependencies, use `libs/onnxruntime-all.jar`.
* To support cross-platform: bundle a single `libs/onnxruntime.jar` and with all of the respective `libs/onnxruntime-jni.jar` and `libs/onnxruntime-lib.jar` for all of the desired OS/Architectures.
* To support use case where an onnxruntime shared library will reside in the system's library search path: bundle a single `libs/onnxruntime.jar` and with all of the `libs/onnxruntime-jni.jar`. The onnxruntime shared library should be loaded using one of the other methods described in the "Advanced Loading" section.

#### Build System Overview 

The main CMake build system delegates building and testing to Gradle.
This allows the CMake system to ensure all of the C/C++ compilation is achieved prior to the Java build.
The Java build depends on C/C++ onnxruntime shared library and a C JNI shared library (source located in the `src/main/native` directory).
The JNI shared library is the glue that allows for Java to call functions in onnxruntime shared library.
Given the fact that CMake injects native dependencies during CMake builds, some gradle tasks (primarily, `build`, `test`, and `check`) may fail.

When running the build script, CMake will compile the `onnxruntime` target and the JNI glue `onnxruntime4j_jni` target and expose the resulting libraries in a place where Gradle can ingest them.
Upon successful compilation of those targets, a special Gradle task to build will be executed. The results will be placed in the `java/build` directory mentioned above.


### Advanced Loading

The default behavior is to load the shared libraries using classpath resources.
If your use case requires custom loading of the shared libraries, please consult the javadoc in the `package-info.java` or `OnnxRuntime.java` files.

## Development

### Code Formatting

[Spotless](https://github.com/diffplug/spotless/tree/master/plugin-gradle) is used to keep the code properly formatted.
Gradle's `spotlessCheck` task will show any misformatted code.
Gradle's `spotlessApply` task will try to fix the formatting.
Misformatted code will raise failures when checks are ran during test run.

###  JNI Headers

When adding or updating native methods in the Java files, it may be necessary to examine the relevant JNI headers in `./build/headers/ai_onnxruntime*.h`.
These files can be manually generated using Gradle's `compileJava` task which will compile the Java and update the header files accordingly.
Then the corresponding C files in `./src/main/native/ai_onnxruntime*.c` may be updated and the build can be ran.

### Dependencies

The Java API does not have any runtime or compile dependencies currently.

