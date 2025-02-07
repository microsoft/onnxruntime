# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set(VCPKG_ENV_PASSTHROUGH_UNTRACKED EMSCRIPTEN_ROOT EMSDK PATH)

if(NOT DEFINED ENV{EMSCRIPTEN_ROOT})
   find_path(EMSCRIPTEN_ROOT "emcc")
else()
   set(EMSCRIPTEN_ROOT "$ENV{EMSCRIPTEN_ROOT}")
endif()

if(NOT EMSCRIPTEN_ROOT)
   if(NOT DEFINED ENV{EMSDK})
      message(FATAL_ERROR "The emcc compiler not found in PATH")
   endif()
   set(EMSCRIPTEN_ROOT "$ENV{EMSDK}/upstream/emscripten")
endif()

if(NOT EXISTS "${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake")
   message(FATAL_ERROR "Emscripten.cmake toolchain file not found")
endif()

set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Emscripten)
set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${EMSCRIPTEN_ROOT}/cmake/Modules/Platform/Emscripten.cmake")
set(VCPKG_TARGET_ARCHITECTURE wasm32)
set(VCPKG_C_FLAGS "-ffunction-sections -fdata-sections -msimd128 -pthread -Wno-pthreads-mem-growth")
set(VCPKG_CXX_FLAGS "/MP /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 /Zc:__cplusplus /GR- /we4541")
set(VCPKG_C_FLAGS_RELEASE "-DNDEBUG -O3")
set(VCPKG_CXX_FLAGS_RELEASE "-DNDEBUG -O3")
set(VCPKG_C_FLAGS_RELWITHDEBINFO "-DNDEBUG -O3")
set(VCPKG_C_FLAGS_RELWITHDEBINFO "-DNDEBUG -O3")
