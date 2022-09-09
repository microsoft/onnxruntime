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
    string(APPEND CMAKE_${lang}_FLAGS_INIT 
        " /D_GAMING_XBOX"
        " /DWINAPI_FAMILY=WINAPI_FAMILY_GAMES"
        " /D_ATL_NO_DEFAULT_LIBS"
        " /D__WRL_NO_DEFAULT_LIB__"
        " /D__WRL_CLASSIC_COM_STRICT__"
        " /D_CRT_USE_WINAPI_PARTITION_APP"
        " /DWIN32_LEAN_AND_MEAN"
        " /favor:AMD64"
    )

    if(GDK_PLATFORM STREQUAL Scarlett)
        string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_GAMING_XBOX_SCARLETT /arch:AVX2")
    elseif(GDK_PLATFORM STREQUAL XboxOne)
        string(APPEND CMAKE_${lang}_FLAGS_INIT " /D_GAMING_XBOX_XBOXONE /arch:AVX")
    endif()

    set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES ${gdk_gxdk_path}/gameKit/Include/${GDK_PLATFORM})

    set(CMAKE_${lang}_STANDARD_LIBRARIES "onecoreuap_apiset.lib" CACHE STRING "" FORCE)
endforeach()

# It's best to avoid inadvertently linking with any libraries not present in the OS.
list(APPEND nodefault_libs 
    advapi32.lib
    comctl32.lib
    comsupp.lib
    dbghelp.lib
    gdi32.lib
    gdiplus.lib
    guardcfw.lib
    kernel32.lib
    mmc.lib
    msimg32.lib
    msvcole.lib
    msvcoled.lib
    mswsock.lib
    ntstrsafe.lib
    ole2.lib
    ole2autd.lib
    ole2auto.lib
    ole2d.lib
    ole2ui.lib
    ole2uid.lib
    ole32.lib
    oleacc.lib
    oleaut32.lib
    oledlg.lib
    oledlgd.lib
    oldnames.lib
    runtimeobject.lib
    shell32.lib
    shlwapi.lib
    strsafe.lib
    urlmon.lib
    user32.lib
    userenv.lib
    wlmole.lib
    wlmoled.lib
    onecore.lib
)

foreach(link_type EXE SHARED MODULE)
    set(CMAKE_${link_type}_LINKER_FLAGS_INIT "")
    foreach(lib ${nodefault_libs})
        string(APPEND CMAKE_${link_type}_LINKER_FLAGS_INIT " /NODEFAULTLIB:${lib}")
    endforeach()
    string(APPEND CMAKE_${link_type}_LINKER_FLAGS_INIT " /DYNAMICBASE /NXCOMPAT /MANIFEST:NO")
endforeach()

set(gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/PIXEvt.lib)
if(GDK_PLATFORM STREQUAL Scarlett)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/Scarlett/d3d12_xs.lib)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/Scarlett/xg_xs.lib)
elseif(GDK_PLATFORM STREQUAL XboxOne)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/XboxOne/d3d12_x.lib)
    list(APPEND gdk_dx_libs ${gdk_gxdk_path}/gameKit/lib/amd64/XboxOne/xg_x.lib)
endif()