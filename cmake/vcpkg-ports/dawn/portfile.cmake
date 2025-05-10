# NOTE: dynamic library vs. static library
#
# The Dawn vcpkg port requires to be built as a shared library, and we are building Dawn as a shared library. However,
# we need to set the `BUILD_SHARED_LIBS` option to `OFF` in this portfile. See the explanation below.
#
# In CMake convention, the `BUILD_SHARED_LIBS` option is used to control whether a library is built as a shared library or a static library.
# However, in the Dawn repository, there are multiple targets. Instead of building each target as a shared library, Dawn
# uses a CMake option `DAWN_BUILD_MONOLITHIC_LIBRARY` to control whether to build a monolithic dynamic library.
#
# When `DAWN_BUILD_MONOLITHIC_LIBRARY` is set to `ON`, a single library is built that contains all the targets. The
# library is always built as a shared library, regardless of the value of `BUILD_SHARED_LIBS`.
#
# In the vcpkg migration, we found that when both `DAWN_BUILD_MONOLITHIC_LIBRARY` and `BUILD_SHARED_LIBS` are set to `ON`, the build process will fail with some unexpected errors.
# So we need to set `BUILD_SHARED_LIBS` to `OFF` in this mode.
#
# The following function call ensures BUILD_SHARED_LIBS is set to OFF.
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(FATAL_ERROR "This port is currently not supported on Emscripten.")
endif()

set(onnxruntime_vcpkg_DAWN_OPTIONS)

list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS

    # enable the vcpkg flag
    -DDAWN_ENABLE_VCPKG=ON

    # fetch dependencies is disabled when using vcpkg
    -DDAWN_FETCH_DEPENDENCIES=OFF

    -DDAWN_BUILD_SAMPLES=OFF
    -DDAWN_ENABLE_NULL=OFF

    -DDAWN_BUILD_TESTS=OFF

    -DDAWN_BUILD_MONOLITHIC_LIBRARY=ON
    -DDAWN_ENABLE_INSTALL=ON

    -DDAWN_ENABLE_DESKTOP_GL=OFF
    -DDAWN_ENABLE_OPENGLES=OFF
    -DDAWN_SUPPORTS_GLFW_FOR_WINDOWING=OFF
    -DDAWN_USE_GLFW=OFF
    -DDAWN_USE_WINDOWS_UI=OFF
    -DTINT_BUILD_GLSL_WRITER=OFF
    -DTINT_BUILD_GLSL_VALIDATOR=OFF

    -DDAWN_DXC_ENABLE_ASSERTS_IN_NDEBUG=OFF
    -DDAWN_USE_X11=OFF

    -DTINT_BUILD_TESTS=OFF
    -DTINT_BUILD_CMD_TOOLS=OFF
    -DTINT_BUILD_IR_BINARY=OFF
    -DTINT_BUILD_SPV_READER=OFF
    -DTINT_BUILD_WGSL_WRITER=ON

    -DDAWN_ENABLE_SPIRV_VALIDATION=OFF

    # explicitly set the jinja2 and markupsafe directories to empty strings
    # when they are empty, the python script will import them from the system
    #
    # pip install jinja2 markupsafe
    #
    -DDAWN_JINJA2_DIR=
    -DDAWN_MARKUPSAFE_DIR=
)

if(VCPKG_TARGET_IS_WINDOWS)
    # feature detection on Windows
    vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
        FEATURES
        windows-use-d3d12 onnxruntime_vcpkg_ENABLE_DAWN_BACKEND_D3D12
        windows-use-vulkan onnxruntime_vcpkg_ENABLE_DAWN_BACKEND_VULKAN
    )

    list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS
        -DDAWN_USE_BUILT_DXC=ON
        -DTINT_BUILD_HLSL_WRITER=ON
    )

    if((NOT onnxruntime_vcpkg_ENABLE_DAWN_BACKEND_VULKAN) AND(NOT onnxruntime_vcpkg_ENABLE_DAWN_BACKEND_D3D12))
        message(FATAL_ERROR "At least one of \"windows-use-d3d12\" or \"windows-use-vulkan\" must be enabled when using Dawn on Windows.")
    endif()

    if(onnxruntime_vcpkg_ENABLE_DAWN_BACKEND_VULKAN)
        list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS
            -DDAWN_ENABLE_VULKAN=ON
            -DTINT_BUILD_SPV_WRITER=ON
        )
    else()
        list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS
            -DDAWN_ENABLE_VULKAN=OFF
        )
    endif()

    if(onnxruntime_vcpkg_ENABLE_DAWN_BACKEND_D3D12)
        list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS
            -DDAWN_ENABLE_D3D12=ON
        )
    else()
        list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS
            -DDAWN_ENABLE_D3D12=OFF
        )
    endif()

    # We are currently always using the D3D12 backend.
    list(APPEND onnxruntime_vcpkg_DAWN_OPTIONS
        -DDAWN_ENABLE_D3D11=OFF
    )
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO fs-eire/dawn
    REF "${VERSION}"
    SHA512 246720afced99f7f50e8d29323d9b5749291f711225c626b42963a81bf9b01512b94f12584cc972fa488f2ee27f6e0167fb370e9421c720e4423be341e236eb4

    # PATCHES
    # dawn.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    WINDOWS_USE_MSBUILD
    OPTIONS
    ${onnxruntime_vcpkg_DAWN_OPTIONS}

    # MAYBE_UNUSED_VARIABLES
)

vcpkg_cmake_install()
