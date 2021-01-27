# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

#set(CMAKE_VERBOSE_MAKEFILE on)

# Setup Java compilation
include(FindJava)
find_package(Java REQUIRED)
include(UseJava)
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_package(JNI REQUIRED)
    include_directories(${JNI_INCLUDE_DIRS})
endif()

set(JAVA_ROOT ${REPO_ROOT}/java)
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java)
if (onnxruntime_RUN_ONNX_TESTS)
  set(JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set(JAVA_DEPENDS onnxruntime)
endif()

# use the gradle wrapper if it exists
if(EXISTS "${JAVA_ROOT}/gradlew")
    set(GRADLE_EXECUTABLE "${JAVA_ROOT}/gradlew")
else()
    # fall back to gradle on our PATH
    find_program(GRADLE_EXECUTABLE gradle)
    if(NOT GRADLE_EXECUTABLE)
        message(SEND_ERROR "Gradle installation not found")
    endif()
endif()
message(STATUS "Using gradle: ${GRADLE_EXECUTABLE}")

# Specify the Java source files
file(GLOB_RECURSE onnxruntime4j_gradle_files "${JAVA_ROOT}/*.gradle")
file(GLOB_RECURSE onnxruntime4j_src "${JAVA_ROOT}/src/main/java/ai/onnxruntime/*.java")
set(JAVA_OUTPUT_JAR ${JAVA_ROOT}/build/libs/onnxruntime.jar)
# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated and the onnxruntime4j_jni target will be rebuilt
set(GRADLE_ARGS clean jar)
if(WIN32)
  set(GRADLE_ARGS ${GRADLE_ARGS} -Dorg.gradle.daemon=false)
endif()
if(onnxruntime_USE_CUDA)
  set(GRADLE_ARGS ${GRADLE_ARGS} -DUSE_CUDA=1)
endif()
add_custom_command(OUTPUT ${JAVA_OUTPUT_JAR} COMMAND ${GRADLE_EXECUTABLE} ${GRADLE_ARGS} WORKING_DIRECTORY ${JAVA_ROOT} DEPENDS ${onnxruntime4j_gradle_files} ${onnxruntime4j_src})
add_custom_target(onnxruntime4j DEPENDS ${JAVA_OUTPUT_JAR})
set_source_files_properties(${JAVA_OUTPUT_JAR} PROPERTIES GENERATED TRUE)
set_property(TARGET onnxruntime4j APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${JAVA_OUTPUT_DIR}")

# Specify the native sources
file(GLOB onnxruntime4j_native_src
    "${JAVA_ROOT}/src/main/native/*.c"
    "${JAVA_ROOT}/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )
# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src})
set_property(TARGET onnxruntime4j_jni PROPERTY CXX_STANDARD 11)

# Tell the JNI code about the requested providers
if (onnxruntime_USE_CUDA)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_CUDA=1)
endif()
if (onnxruntime_USE_DNNL)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_DNNL=1)
endif()
if (onnxruntime_USE_OPENVINO)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_OPENVINO=1)
endif()
if (onnxruntime_USE_TENSORRT)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_TENSORRT=1)
endif()
if (onnxruntime_USE_NNAPI_BUILTIN)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_NNAPI=1)
endif()
if (onnxruntime_USE_NUPHAR)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_NUPHAR=1)
endif()
if (onnxruntime_USE_ACL)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_ACL=1)
endif()
if (onnxruntime_USE_DML)
  target_compile_definitions(onnxruntime4j_jni PRIVATE USE_DIRECTML=1)
endif()

# depend on java sources. if they change, the JNI should recompile
add_dependencies(onnxruntime4j_jni onnxruntime4j)
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
# the JNI headers are generated in the onnxruntime4j target
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${JAVA_ROOT}/build/headers)
target_link_libraries(onnxruntime4j_jni PUBLIC onnxruntime)

set(JAVA_PACKAGE_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build)
file(MAKE_DIRECTORY ${JAVA_PACKAGE_OUTPUT_DIR})
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(ANDROID_PACKAGE_OUTPUT_DIR ${JAVA_PACKAGE_OUTPUT_DIR}/android)
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_OUTPUT_DIR})
endif()

# Set platform and arch for packaging
# Checks the names set by MLAS on non-Windows platforms first
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(JNI_ARCH ${ANDROID_ABI})
elseif (ARM64)
  set(JNI_ARCH aarch64)
elseif (X86_64)
  set(JNI_ARCH x64)
elseif (POWER)
  set(JNI_ARCH ppc64)
else()
  # Now mirror the checks used with MSVC
  if(MSVC)
    if(onnxruntime_target_platform STREQUAL "ARM64")
      set(JNI_ARCH aarch64)
    elseif(onnxruntime_target_platform STREQUAL "x64")
      set(JNI_ARCH x64)
    else()
      # if everything else failed then we're on a 32-bit arch and Java isn't supported
      message(FATAL_ERROR "Java is currently not supported on 32-bit x86 architecture")
    endif()
  else()
    # if everything else failed then we're on a 32-bit arch and Java isn't supported
    message(FATAL_ERROR "Java is currently not supported on 32-bit x86 architecture")
  endif()
endif()

if (WIN32)
  set(JAVA_PLAT "win")
elseif (APPLE)
  set(JAVA_PLAT "osx")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(JAVA_PLAT "linux")
else()
  # We don't do distribution for Android
  # Set for completeness
  set(JAVA_PLAT "android")
endif()

# Similar to Nuget schema
set(JAVA_OS_ARCH ${JAVA_PLAT}-${JNI_ARCH})

# expose native libraries to the gradle build process
set(JAVA_PACKAGE_DIR ai/onnxruntime/native/${JAVA_OS_ARCH})
set(JAVA_NATIVE_LIB_DIR ${JAVA_OUTPUT_DIR}/native-lib)
set(JAVA_NATIVE_JNI_DIR ${JAVA_OUTPUT_DIR}/native-jni)
set(JAVA_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/${JAVA_PACKAGE_DIR})
set(JAVA_PACKAGE_JNI_DIR ${JAVA_NATIVE_JNI_DIR}/${JAVA_PACKAGE_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_LIB_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_JNI_DIR})

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(ANDROID_PACKAGE_JNILIBS_DIR ${JAVA_OUTPUT_DIR}/android)
  set(ANDROID_PACKAGE_ABI_DIR ${ANDROID_PACKAGE_JNILIBS_DIR}/${ANDROID_ABI})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_JNILIBS_DIR})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_ABI_DIR})
endif()

# On Windows TARGET_LINKER_FILE_NAME is the .lib, TARGET_FILE_NAME is the .dll
if (WIN32)
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime>)
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_FILE_NAME:onnxruntime4j_jni>)
else()
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime>)
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni>)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime> ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime>)
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime4j_jni> ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni>)
endif()

# run the build process (this copies the results back into CMAKE_CURRENT_BINARY_DIR)
set(GRADLE_ARGS cmakeBuild -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR})
if(WIN32)
  set(GRADLE_ARGS ${GRADLE_ARGS} -Dorg.gradle.daemon=false)
endif()
if(onnxruntime_USE_CUDA)
  set(GRADLE_ARGS ${GRADLE_ARGS} -DUSE_CUDA=1)
endif()
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} ${GRADLE_ARGS} WORKING_DIRECTORY ${JAVA_ROOT})
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} -b build-android.gradle -c settings-android.gradle build -DjniLibsDir=${ANDROID_PACKAGE_JNILIBS_DIR} -DbuildDir=${ANDROID_PACKAGE_OUTPUT_DIR} WORKING_DIRECTORY ${JAVA_ROOT})
endif()
