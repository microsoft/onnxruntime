# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_C_FLAGS "-g -ffunction-sections -fdata-sections -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0")
set(VCPKG_CXX_FLAGS "-g -ffunction-sections -fdata-sections -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0 -fno-rtti -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables")
set(VCPKG_C_FLAGS_RELEASE "-DNDEBUG -O3 -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong")
set(VCPKG_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong")
set(VCPKG_C_FLAGS_RELWITHDEBINFO "-DNDEBUG -O3 -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong")
set(VCPKG_C_FLAGS_RELWITHDEBINFO "-DNDEBUG -O3 -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong")
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS --compile-no-warning-as-error -DBENCHMARK_ENABLE_WERROR=OFF)
set(VCPKG_LINKER_FLAGS "-Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -g")
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_CXX_STANDARD=17)
if(PORT MATCHES "onnx")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_DISABLE_STATIC_REGISTRATION=ON"
    )
endif()
if(PORT MATCHES "benchmark")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DBENCHMARK_ENABLE_WERROR=OFF"
    )
endif()
if(PORT MATCHES "onnx")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_DISABLE_EXCEPTIONS=ON"
    )
endif()
