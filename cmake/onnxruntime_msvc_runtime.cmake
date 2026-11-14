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

# Returns the MSVC tools root (VC/Tools/MSVC/<version>) that provides the MSVC runtime libraries.
function(onnxruntime_get_msvc_tools_root result_var)
  if (NOT (CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
    # CMAKE_C_COMPILER is <MSVC tools root>/bin/Host<arch>/<arch>/cl.exe.
    cmake_path(SET msvc_tools_root NORMALIZE "${CMAKE_C_COMPILER}/../../../..")
    set(${result_var} "${msvc_tools_root}" PARENT_SCOPE)
    return()
  endif()

  # clang-cl lives under VC/Tools/Llvm, so the runtime cannot be found relative to it. The VS ClangCL toolset links
  # the MSVC runtime of the MSVC tools version MSBuild selects: the generator toolset's version= field if given,
  # otherwise the instance's default.
  if (NOT CMAKE_GENERATOR_INSTANCE)
    message(FATAL_ERROR "Linking the OneCore MSVC runtime with clang-cl requires a Visual Studio generator.")
  endif()
  set(msvc_tools_dir "${CMAKE_GENERATOR_INSTANCE}/VC/Tools/MSVC")
  if (CMAKE_VS_PLATFORM_TOOLSET_VERSION)
    file(GLOB msvc_tools_versions LIST_DIRECTORIES true RELATIVE "${msvc_tools_dir}"
         "${msvc_tools_dir}/${CMAKE_VS_PLATFORM_TOOLSET_VERSION}*")
    list(SORT msvc_tools_versions COMPARE NATURAL ORDER DESCENDING)
    if (msvc_tools_versions)
      list(GET msvc_tools_versions 0 msvc_tools_version)
    endif()
  else()
    set(msvc_tools_version_file "${CMAKE_GENERATOR_INSTANCE}/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt")
    if (EXISTS "${msvc_tools_version_file}")
      file(STRINGS "${msvc_tools_version_file}" msvc_tools_version LIMIT_COUNT 1)
      string(STRIP "${msvc_tools_version}" msvc_tools_version)
    endif()
  endif()
  if (NOT msvc_tools_version OR NOT IS_DIRECTORY "${msvc_tools_dir}/${msvc_tools_version}")
    message(FATAL_ERROR "Unable to locate the MSVC tools for clang-cl under ${msvc_tools_dir} "
                        "(requested version: '${CMAKE_VS_PLATFORM_TOOLSET_VERSION}').")
  endif()
  set(${result_var} "${msvc_tools_dir}/${msvc_tools_version}" PARENT_SCOPE)
endfunction()

function(onnxruntime_configure_msvc_onecore_runtime)
  # GDK builds are skipped because gdk_toolchain.cmake configures the runtime, WINAPI_FAMILY and related flags.
  if (onnxruntime_BUILD_FOR_ONECORE AND NOT GDK_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
    # OneCore builds must link the OneCore flavor of the MSVC runtime. Its lib directory is prepended to the
    # linker search path below so it takes precedence over the desktop runtime libraries found via LIB.
    onnxruntime_get_msvc_tools_root(msvc_tools_root)
    set(msvc_lib_root "${msvc_tools_root}/lib")
    set(msvc_onecore_platform "${onnxruntime_target_platform}")
    if (msvc_onecore_platform STREQUAL "ARM64EC")
      # MSVC ships no ARM64EC runtime directory; ARM64EC uses the ARM64 hybrid CRT libraries.
      set(msvc_onecore_platform "ARM64")
    endif()
    set(msvc_onecore_lib_dir "${msvc_lib_root}/onecore/${msvc_onecore_platform}")

    onnxruntime_msvc_spectre_enabled("${CMAKE_C_FLAGS}" msvc_c_spectre_enabled)
    onnxruntime_msvc_spectre_enabled("${CMAKE_CXX_FLAGS}" msvc_cxx_spectre_enabled)
    # onnxruntime_USE_MSVC_SPECTRE_RUNTIME requests the Spectre-mitigated runtime for compilers such as clang-cl
    # that do not implement /Qspectre.
    if (msvc_c_spectre_enabled OR msvc_cxx_spectre_enabled OR onnxruntime_USE_MSVC_SPECTRE_RUNTIME)
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
