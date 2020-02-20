# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.0)

# utility
function(convert_forward_slashes_to_back input output)
    string(REGEX REPLACE "/" "\\\\" backwards ${input})
    set(${output} ${backwards} PARENT_SCOPE)
endfunction()

# get window 10 install path from registry
function(get_installed_sdk
    sdk_folder        # the current sdk folder
    output_sdk_version    # the current sdk version
)
    # return the kit path
    get_filename_component(win10_sdk_root "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE CACHE)
    set(${sdk_folder} ${win10_sdk_root} PARENT_SCOPE)

    # return the sdk version
    if(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION)
        set(${output_sdk_version} ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION} PARENT_SCOPE)
    else()
        # choose the SDK matching the system version, or fallback to the latest
        file(GLOB win10_sdks RELATIVE "${win10_sdk_root}/UnionMetadata" "${win10_sdk_root}/UnionMetadata/*.*.*.*")
        list(GET win10_sdks 0 latest_sdk)
        foreach(sdk IN LISTS win10_sdks)
            string(FIND ${sdk} ${CMAKE_SYSTEM_VERSION} is_system_version)
            if(NOT ${is_system_version} EQUAL -1)
                set(${output_sdk_version} ${sdk} PARENT_SCOPE)
                return()
            elseif(sdk VERSION_GREATER latest_sdk)
                set(latest_sdk ${sdk})
            endif()
        endforeach()
        set(${output_sdk_version} ${latest_sdk} PARENT_SCOPE)
    endif()
endfunction()

# current sdk binary directory
function(get_sdk_binary_directory
    sdk_folder        # the kit path
    sdk_version       # the sdk version
    binary_dir        # the output folder variable
)
    set(${binary_dir} "${sdk_folder}/bin/${sdk_version}" PARENT_SCOPE)
endfunction()

# current sdk include directory
function(get_sdk_include_folder
    sdk_folder        # the kit path
    sdk_version       # the sdk version
    include_dir       # the output folder variable
)
    set(${include_dir} "${sdk_folder}/include/${sdk_version}" PARENT_SCOPE)
endfunction()

# current sdk metadata directory
function(get_sdk_metadata_folder
    sdk_folder        # the kit path
    sdk_version       # the sdk version
    metadata_dir      # the output folder variable
)
    set(${metadata_dir} "${sdk_folder}/UnionMetadata/${sdk_version}" PARENT_SCOPE)
endfunction()

# current sdk midl exe path
function(get_sdk_midl_exe
    sdk_folder        # the kit path
    sdk_version       # the sdk version
    midl_exe_path     # the output exe path
)
    get_sdk_binary_directory(${sdk_folder} ${sdk_version} bin_dir)
    set(${midl_exe_path} "${bin_dir}/x64/midlrt.exe" PARENT_SCOPE)
endfunction()

# current cppwinrt cppwinrt exe path
function(get_installed_sdk_cppwinrt_exe
    sdk_folder        # the kit path
    sdk_version       # the sdk version
    cppwinrt_exe_path # the output exe path
)
    get_sdk_binary_directory(${sdk_folder} ${sdk_version} bin_dir)
    set(${cppwinrt_exe_path} "${bin_dir}/x64/cppwinrt.exe" PARENT_SCOPE)
endfunction()

# current cppwinrt cppwinrt exe path
function(get_sdk_cppwinrt_exe
    sdk_folder        # the kit path
    sdk_version       # the sdk version
    output_cppwinrt_exe_path # the output exe path
)
  if (NOT DEFINED winml_CPPWINRT_EXE_PATH_OVERRIDE)
    get_installed_sdk_cppwinrt_exe(${sdk_folder} ${sdk_version} cppwinrt_exe_path)
    set(${output_cppwinrt_exe_path} ${cppwinrt_exe_path} PARENT_SCOPE)
  else ()
    set(${output_cppwinrt_exe_path} ${winml_CPPWINRT_EXE_PATH_OVERRIDE} PARENT_SCOPE)
  endif()
endfunction()

function(get_sdk
    output_sdk_folder     # the path to the current sdk kit folder
    output_sdk_version    # the current sdk version
)
  if ((NOT DEFINED winml_WINDOWS_SDK_DIR_OVERRIDE) AND
      (NOT DEFINED winml_WINDOWS_SDK_VERSION_OVERRIDE))
    get_installed_sdk(sdk_folder sdk_version)
    set(${output_sdk_folder} ${sdk_folder} PARENT_SCOPE)
    set(${output_sdk_version} ${sdk_version} PARENT_SCOPE)
  elseif ((DEFINED winml_WINDOWS_SDK_DIR_OVERRIDE) AND
          (DEFINED winml_WINDOWS_SDK_VERSION_OVERRIDE))
    set(${output_sdk_folder} ${winml_WINDOWS_SDK_DIR_OVERRIDE} PARENT_SCOPE)
    set(${output_sdk_version} ${winml_WINDOWS_SDK_VERSION_OVERRIDE} PARENT_SCOPE)
  else()
    message(
      FATAL_ERROR
      "Options winml_WINDOWS_SDK_DIR_OVERRIDE and winml_WINDOWS_SDK_VERSION_OVERRIDE must be defined together, or not at all.")
  endif()
endfunction()
