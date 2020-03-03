# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# The default libraries to link with in Windows are Win32 libs:
# kernel32.lib;user32.lib;gdi32.lib;winspool.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;comdlg32.lib;advapi32.lib
# If we're targeting WCOS, we must override them with a WCOS umbrella library.
# Else, onnxruntime must use the Win32 libs to support downlevel platforms, and WinML must use the umbrella library.
# CMake doesn't support multiple toolchains in the same project, so in this case we use a small hack clearing
# CMAKE_CXX_STANDARD_LIBRARIES and applying the Win32 or WCOS libraries to each target manually.

foreach(default_lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdgl32.lib advapi32.lib)
  set(wcos_options "${wcos_options} /NODEFAULTLIB:${default_lib}")
endforeach()

if (onnxruntime_ENABLE_WCOS)
    set(CMAKE_CXX_STANDARD_LIBRARIES "${wcos_options} windowsapp.lib")
    return()
endif()

set(win32_libs "${CMAKE_CXX_STANDARD_LIBRARIES}")
set(CMAKE_CXX_STANDARD_LIBRARIES "")
separate_arguments(win32_libs)
separate_arguments(wcos_options)
get_directory_property(all_targets BUILDSYSTEM_TARGETS)
foreach(target ${all_targets})
    get_target_property(target_type ${target} TYPE)
    if (NOT (target_type STREQUAL INTERFACE_LIBRARY OR target_type STREQUAL UTILITY))
        get_target_property(target_language ${target} LINKER_LANGUAGE)
        if (target_language STREQUAL CXX OR target_language MATCHES NOTFOUND)
            if (target MATCHES "winml_")
                target_link_libraries(${target} PRIVATE windowsapp.lib)
                target_link_options(${target} PRIVATE ${wcos_options})
            else()
                target_link_libraries(${target} PRIVATE ${win32_libs})
            endif()
        endif()
    endif()
endforeach()
