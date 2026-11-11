# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# CMake's default MSVC system libraries include kernel32.lib; OneCore builds replace them with umbrella libraries.
function(onnxruntime_detect_onecore_build standard_libraries result_var)
  if (WIN32 AND NOT standard_libraries MATCHES kernel32.lib)
    set(${result_var} ON PARENT_SCOPE)
  else()
    set(${result_var} OFF PARENT_SCOPE)
  endif()
endfunction()

# Recognizes the switches BinSkim BA2024 treats as Spectre mitigations (/Qspectre, /Qspectre-load,
# /Qspectre-load-cf, with either / or - prefix). The last enabling or disabling switch wins.
function(onnxruntime_msvc_spectre_enabled flags result_var)
  separate_arguments(msvc_flags WINDOWS_COMMAND "${flags}")
  set(spectre_enabled OFF)
  foreach(flag IN LISTS msvc_flags)
    if (flag MATCHES "^[/-]Qspectre(-load(-cf)?)?$")
      set(spectre_enabled ON)
    elseif(flag MATCHES "^[/-]Qspectre(-load(-cf)?)?-$")
      set(spectre_enabled OFF)
    endif()
  endforeach()
  set(${result_var} ${spectre_enabled} PARENT_SCOPE)
endfunction()

function(onnxruntime_configure_msvc_onecore_runtime)
  # GDK builds are skipped because gdk_toolchain.cmake configures the runtime, WINAPI_FAMILY and related flags.
  if (onnxruntime_BUILD_FOR_ONECORE AND NOT GDK_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
    # OneCore builds must link the OneCore flavor of the MSVC runtime. Its lib directory is prepended to the
    # linker search path below so it takes precedence over the desktop runtime libraries found via LIB.
    # CMAKE_C_COMPILER is <MSVC tools root>/bin/Host<arch>/<arch>/cl.exe.
    cmake_path(SET msvc_lib_root NORMALIZE "${CMAKE_C_COMPILER}/../../../../lib")
    set(msvc_onecore_platform "${onnxruntime_target_platform}")
    if (msvc_onecore_platform STREQUAL "ARM64EC")
      # MSVC ships no ARM64EC runtime directory; ARM64EC uses the ARM64 hybrid CRT libraries.
      set(msvc_onecore_platform "ARM64")
    endif()
    set(msvc_onecore_lib_dir "${msvc_lib_root}/onecore/${msvc_onecore_platform}")

    onnxruntime_msvc_spectre_enabled("${CMAKE_C_FLAGS}" msvc_c_spectre_enabled)
    onnxruntime_msvc_spectre_enabled("${CMAKE_CXX_FLAGS}" msvc_cxx_spectre_enabled)
    if (msvc_c_spectre_enabled OR msvc_cxx_spectre_enabled)
      set(msvc_onecore_lib_dir "${msvc_lib_root}/spectre/onecore/${msvc_onecore_platform}")
      # Only the release libraries are required: debug CRT libraries are not Spectre-mitigated by design, and
      # BinSkim BA2024 skips modules that link them.
      foreach(msvc_runtime_lib libcmt.lib libcpmt.lib libvcruntime.lib)
        if (NOT EXISTS "${msvc_onecore_lib_dir}/${msvc_runtime_lib}")
          message(FATAL_ERROR
                  "Required Spectre-mitigated OneCore runtime library is missing: "
                  "${msvc_onecore_lib_dir}/${msvc_runtime_lib}\n"
                  "Spectre mitigation is enabled, so the Visual Studio 'MSVC <version> Libs for Spectre' component "
                  "for this target architecture must be installed (or the Spectre mitigation flag removed).")
        endif()
      endforeach()
    endif()

    if (NOT IS_DIRECTORY "${msvc_onecore_lib_dir}")
      message(FATAL_ERROR "OneCore MSVC runtime library directory is missing: ${msvc_onecore_lib_dir}")
    endif()
    message(STATUS "MSVC OneCore runtime library directory: ${msvc_onecore_lib_dir}")
    link_directories(BEFORE "${msvc_onecore_lib_dir}")
    # The MSVC runtime libraries contain a DEFAULTLIB entry for onecore.lib, but it does not conflict with
    # onecoreuap.lib.
  endif()
endfunction()
