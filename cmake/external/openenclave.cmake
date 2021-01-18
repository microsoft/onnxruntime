# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Open Enclave doesn't provide compiler runtime support functions.
# We use compiler-rt which is compatible with gcc and clang.
# Not linking against it would lead to unresolved symbols for __udivti3 and others.
# See https://github.com/Microsoft/openenclave/issues/1729.
enable_language(ASM)
set(COMPILER_RT_DEFAULT_TARGET_TRIPLE x86_64-pc-linux-unknown CACHE STRING "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/external/compiler-rt EXCLUDE_FROM_ALL)
add_library(compiler-rt-builtins ALIAS clang_rt.builtins-x86_64)

if (NOT TARGET openenclave::oeedger8r)
    find_package(openenclave REQUIRED CONFIG)
    message(STATUS "Using Open Enclave ${openenclave_VERSION} from ${openenclave_CONFIG}")
endif()

add_library(openenclave-host INTERFACE IMPORTED)
target_link_libraries(openenclave-host INTERFACE
    openenclave::oehostapp
)

add_library(openenclave-enclave INTERFACE IMPORTED)
target_link_libraries(openenclave-enclave INTERFACE
    openenclave::oeenclave
    openenclave::oelibcxx
    compiler-rt-builtins
)

# Obtain default compiler include directory to gain access to intrinsics and other
# headers like cpuid.h.
execute_process(
    COMMAND /bin/bash ${CMAKE_CURRENT_LIST_DIR}/openenclave_get_c_compiler_inc_dir.sh ${CMAKE_C_COMPILER}
    OUTPUT_VARIABLE ONNXRUNTIME_C_COMPILER_INC
    ERROR_VARIABLE ONNXRUNTIME_ERR
)
if (NOT ONNXRUNTIME_ERR STREQUAL "")
    message(FATAL_ERROR ${ONNXRUNTIME_ERR})
endif()

# Works around https://gitlab.kitware.com/cmake/cmake/issues/19227#note_570839.
if (onnxruntime_OPENENCLAVE_BUILD_ENCLAVE)
    unset(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    unset(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
endif()
set(ONNXRUNTIME_C_COMPILER_INC ${ONNXRUNTIME_C_COMPILER_INC})

# Open Enclave lacks some non-standard headers that are typically
# expected to exist, so let's add them here.
target_include_directories(openenclave-enclave INTERFACE
    ${PROJECT_SOURCE_DIR}/patches/openenclave/libcxxrt
    )

# Open Enclave lacks some functions from the standard library,
# so let's stub them out as best we can.
target_sources(openenclave-enclave INTERFACE
    ${PROJECT_SOURCE_DIR}/patches/openenclave/stubs_undefined.c
    ${PROJECT_SOURCE_DIR}/patches/openenclave/stubs_override.c
    )

# The compiler include dir must come *after* Open Enclave's standard library folders,
# otherwise there will be type re-definitions. This is not possible when
# adding to openenclave-enclave, so let's add to the existing target instead.
target_include_directories(openenclave::oelibc_includes SYSTEM INTERFACE
    ${ONNXRUNTIME_C_COMPILER_INC}
    )

# Allow multiple definitions so that stubs_override.c can be used.
target_link_options(openenclave-enclave INTERFACE
    "LINKER:-z,muldefs"
    )

# Open Enclave marks unsupported standard library functions as "deprecated".
# We provide stubs for these, so let's silence these warnings.
target_compile_definitions(openenclave-enclave INTERFACE
    OE_LIBC_SUPPRESS_DEPRECATIONS
    )

# Provide a way to #ifdef unsupported functionality.
target_compile_definitions(openenclave-enclave INTERFACE
    OPENENCLAVE_BUILD_ENCLAVE
    )