# ONNX Runtime Java API

This directory contains the Java language binding for the ONNX runtime.
Java Native Interface (JNI) is used to allow for seamless calls to ONNX runtime from Java.

## Usage

This document pertains to developing, building, running, and testing the API itself in your local environment.
For general purpose usage of the publicly distributed API, please see the [general Java API documentation](https://www.onnxruntime.ai/docs/reference/api/java-api.html).

### Building

Use the main project's [build instructions](https://www.onnxruntime.ai/docs/how-to/build.html) with the `--build_java` option.

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
* `libs/onnxruntime-VERSION.jar` - JAR with compiled classes, platform-specific JNI shared library, and platform-specific onnxruntime shared library.

#### Build System Overview 

The main CMake build system delegates building and testing to Gradle.
This allows the CMake system to ensure all of the C/C++ compilation is achieved prior to the Java build.
The Java build depends on C/C++ onnxruntime shared library and a C JNI shared library (source located in the `src/main/native` directory).
The JNI shared library is the glue that allows for Java to call functions in onnxruntime shared library.
Given the fact that CMake injects native dependencies during CMake builds, some gradle tasks (primarily, `build`, `test`, and `check`) may fail.

When running the build script, CMake will compile the `onnxruntime` target and the JNI glue `onnxruntime4j_jni` target and expose the resulting libraries in a place where Gradle can ingest them.
Upon successful compilation of those targets, a special Gradle task to build will be executed. The results will be placed in the output directory stated above.

### Advanced Loading

The default behavior is to load the shared libraries using classpath resources.
If your use case requires custom loading of the shared libraries, please consult the javadoc in the [package-info.java](src/main/java/ai/onnxruntime/package-info.java) or [OnnxRuntime.java](src/main/java/ai/onnxruntime/OnnxRuntime.java) files.

## Development

### Code Formatting

[Spotless](https://github.com/diffplug/spotless/tree/master/plugin-gradle) is used to keep the code properly formatted.
Gradle's `spotlessCheck` task will show any misformatted code.
Gradle's `spotlessApply` task will try to fix the formatting.
Misformatted code will raise failures when checks are ran during test run.

###  JNI Headers

When adding or updating native methods in the Java files, it may be necessary to examine the relevant JNI headers in `build/headers/ai_onnxruntime*.h`.
These files can be manually generated using Gradle's `compileJava` task which will compile the Java and update the header files accordingly.
Then the corresponding C files in `./src/main/native/ai_onnxruntime*.c` may be updated and the build can be ran.

### Dependencies

The Java API does not have any runtime or compile dependencies currently.
