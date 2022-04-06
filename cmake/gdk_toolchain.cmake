# Options to configure the toolchain.
set(GDK_EDITION 210602 CACHE STRING "GDK edition.")
set(GDK_PLATFORM Scarlett CACHE STRING "GDK target platform.")

# Required to propagate variables when CMake calls try_compile() to test the toolchain.
# https://cmake.org/cmake/help/latest/variable/CMAKE_TRY_COMPILE_PLATFORM_VARIABLES.html
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES GDK_EDITION GDK_PLATFORM)

cmake_path(SET gdk_gxdk_path $ENV{GameDK}/${GDK_EDITION}/GXDK NORMALIZE)

# Set C/C++ compile flags and additional include directories.
foreach(lang C CXX)
    set(CMAKE_${lang}_FLAGS_INIT "")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_GAMING_XBOX")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /DWINAPI_FAMILY=WINAPI_FAMILY_GAMES")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_ATL_NO_DEFAULT_LIBS")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /D__WRL_NO_DEFAULT_LIB__")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /D__WRL_CLASSIC_COM_STRICT__")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_CRT_USE_WINAPI_PARTITION_APP")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /DWIN32_LEAN_AND_MEAN")
    string(APPEND CMAKE_${lang}_FLAGS_INIT " /favor:AMD64")

    if(GDK_PLATFORM STREQUAL Scarlett)
        string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_GAMING_XBOX_SCARLETT")
        string(APPEND CMAKE_${lang}_FLAGS_INIT " /arch:AVX2")
    elseif(GDK_PLATFORM STREQUAL XboxOne)
        string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_GAMING_XBOX_XBOXONE")
        string(APPEND CMAKE_${lang}_FLAGS_INIT " /arch:AVX")
    endif()

    set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES ${gdk_gxdk_path}/gameKit/Include/${GDK_PLATFORM})

    set(CMAKE_${lang}_STANDARD_LIBRARIES "onecoreuap_apiset.lib" CACHE STRING "" FORCE)
endforeach()

# Workaround for std::getenv only being defined under _CRT_USE_WINAPI_FAMILY_DESKTOP_APP.
set(gdk_workaround_h ${CMAKE_BINARY_DIR}/gdk_workarounds.h)
file(WRITE ${gdk_workaround_h} [[
#pragma once
#include <cstdlib>
namespace std { using ::getenv; }
]])
string(APPEND CMAKE_CXX_FLAGS_INIT " /FI${gdk_workaround_h}")

# It's best to avoid inadvertently linking with any libraries not present in the OS.
set(nodefault_libs "")
list(APPEND nodefault_libs advapi32.lib)
list(APPEND nodefault_libs comctl32.lib)
list(APPEND nodefault_libs comsupp.lib)
list(APPEND nodefault_libs dbghelp.lib)
list(APPEND nodefault_libs gdi32.lib)
list(APPEND nodefault_libs gdiplus.lib)
list(APPEND nodefault_libs guardcfw.lib)
list(APPEND nodefault_libs kernel32.lib)
list(APPEND nodefault_libs mmc.lib)
list(APPEND nodefault_libs msimg32.lib)
list(APPEND nodefault_libs msvcole.lib)
list(APPEND nodefault_libs msvcoled.lib)
list(APPEND nodefault_libs mswsock.lib)
list(APPEND nodefault_libs ntstrsafe.lib)
list(APPEND nodefault_libs ole2.lib)
list(APPEND nodefault_libs ole2autd.lib)
list(APPEND nodefault_libs ole2auto.lib)
list(APPEND nodefault_libs ole2d.lib)
list(APPEND nodefault_libs ole2ui.lib)
list(APPEND nodefault_libs ole2uid.lib)
list(APPEND nodefault_libs ole32.lib)
list(APPEND nodefault_libs oleacc.lib)
list(APPEND nodefault_libs oleaut32.lib)
list(APPEND nodefault_libs oledlg.lib)
list(APPEND nodefault_libs oledlgd.lib)
list(APPEND nodefault_libs oldnames.lib)
list(APPEND nodefault_libs runtimeobject.lib)
list(APPEND nodefault_libs shell32.lib)
list(APPEND nodefault_libs shlwapi.lib)
list(APPEND nodefault_libs strsafe.lib)
list(APPEND nodefault_libs urlmon.lib)
list(APPEND nodefault_libs user32.lib)
list(APPEND nodefault_libs userenv.lib)
list(APPEND nodefault_libs wlmole.lib)
list(APPEND nodefault_libs wlmoled.lib)
list(APPEND nodefault_libs onecore.lib)

foreach(link_type EXE SHARED MODULE)
    set(CMAKE_${link_type}_LINKER_FLAGS_INIT "")
    foreach(lib ${nodefault_libs})
        string(APPEND CMAKE_${link_type}_LINKER_FLAGS_INIT " /NODEFAULTLIB:${lib}")
    endforeach()
    string(APPEND CMAKE_${link_type}_LINKER_FLAGS_INIT " /DYNAMICBASE")
    string(APPEND CMAKE_${link_type}_LINKER_FLAGS_INIT " /NXCOMPAT")
    string(APPEND CMAKE_${link_type}_LINKER_FLAGS_INIT " /MANIFEST:NO")
endforeach()

set(gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/PIXEvt.lib)
if(GDK_PLATFORM STREQUAL Scarlett)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/Scarlett/d3d12_xs.lib)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/Scarlett/xg_xs.lib)
elseif(GDK_PLATFORM STREQUAL XboxOne)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/XboxOne/d3d12_x.lib)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/XboxOne/xg_x.lib)
endif()