import os
from collections.abc import Iterable
from pathlib import Path

# The official vcpkg repository has about 80 different triplets. But ONNX Runtime has many more build variants. For example, in general, for each platform, we need to support builds with C++ exceptions, builds without C++ exceptions, builds with C++ RTTI, builds without C++ RTTI, linking to static C++ runtime, linking to dynamic (shared) C++ runtime, builds with address sanitizer, builds without address sanitizer, etc. Therefore, this script file was created to dynamically generate the triplet files on-the-fly.

# Originally, we tried to check in all the generated files into our repository so that people could build onnxruntime without using build.py or any other Python scripts in the "/tools" directory. However, we encountered an issue when adding support for WASM builds. VCPKG has a limitation that when doing cross-compiling, the triplet file must specify the full path of the chain-loaded toolchain file. The file needs to be located either via environment variables (like ANDROID_NDK_HOME) or via an absolute path. Since environment variables are hard to track, we chose the latter approach. So the generated triplet files may contain absolute file paths that are only valid on the current build machine.

# The compiler flags(CFLAGS/CXXFLAGS/LDFLAGS) settings in this file must be consistent with the cmake code in "cmake/adjust_global_compile_flags.cmake" so that all the statically linked code were compiled by the same set of compile flags.


# This is a way to add customizations to the official VCPKG ports.
def add_port_configs(f, has_exception: bool, is_emscripten: bool, enable_minimal_build: bool) -> None:
    """
    Add port-specific configurations to the triplet file.

    Args:
        f (file object): The file object to write configurations.
        has_exception (bool): Flag indicating if exceptions are enabled.
        is_emscripten (bool): Flag indicating if the target is Emscripten.
        enable_minimal_build (bool): Flag indicating if ONNX minimal build is enabled.
    """
    f.write(
        r"""if(PORT MATCHES "benchmark")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DBENCHMARK_ENABLE_WERROR=OFF"
    )
endif()
"""
    )
    f.write(
        r"""if(PORT MATCHES "date")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DENABLE_DATE_TESTING=OFF"
        "-DBUILD_TZ_LIB=OFF"
        "-DUSE_SYSTEM_TZ_DB=ON"
    )
endif()
"""
    )
    if is_emscripten:
        f.write(
            r"""if(PORT MATCHES "gtest")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-Dgtest_disable_pthreads=ON"
    )
endif()
"""
        )

    # Add ONNX specific flags based on exception and minimal build settings
    f.write(r"""if(PORT MATCHES "onnx")""")  # Start ONNX-specific block
    f.write(r"""
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_DISABLE_STATIC_REGISTRATION=ON"
    )
    """)
    if not has_exception:
        # From the ORT CMake logic: onnxruntime_DISABLE_EXCEPTIONS requires onnxruntime_MINIMAL_BUILD.
        # While we add the flag here based on has_exception, the calling build script
        # must ensure it only uses a noexception triplet when a minimal build is intended for ORT itself.
        # This triplet setting makes sure the ONNX *dependency* is built correctly if no-exception is requested.
        f.write(
            r"""
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_DISABLE_EXCEPTIONS=ON"
    )"""
        )

    if enable_minimal_build:
        f.write(
            r"""
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_MINIMAL_BUILD=ON"
    )"""
        )

    f.write(r"""
endif() # End ONNX-specific block
""")


def add_copyright_header(f) -> None:
    """
    Add copyright header to the triplet file.

    Args:
        f (file object): The file object to write the header.
    """
    f.write(
        r"""# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
    )


def add_build_type(f, build_type: str) -> None:
    """
    Add build type to the triplet file.

    Args:
        f (file object): The file object to write the build type.
        build_type (str): The build type to add. Must be one of "Debug", "Release", "RelWithDebInfo", or "MinSizeRel".
    """
    if build_type not in ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"]:
        raise ValueError(
            f"Invalid build type: {build_type}. Must be one of 'Debug', 'Release', 'RelWithDebInfo', or 'MinSizeRel'."
        )

    f.write(
        f"""set(VCPKG_BUILD_TYPE {"debug" if build_type == "Debug" else "release"})
"""
    )


def generate_triplet_for_android(
    build_dir: str,
    configs: Iterable[str],
    target_abi: str,
    enable_rtti: bool,
    enable_exception: bool,
    enable_asan: bool,
    enable_minimal_build: bool,
    use_cpp_shared: bool,
    android_api_level: int,
) -> None:
    """
    Generate triplet file for Android platform.

    Args:
        build_dir (str): The directory to save the generated triplet files.
        target_abi (str): The target ABI.
        enable_rtti (bool): Flag indicating if RTTI is enabled.
        enable_exception (bool): Flag indicating if exceptions are enabled.
        enable_asan (bool): Flag indicating if AddressSanitizer is enabled.
        enable_minimal_build (bool): Flag indicating if ONNX minimal build is enabled.
        use_cpp_shared(bool): The type of C++ Runtime to use. If it is false, use "c++_static" which is the default for most CMake projects. Otherwise set the runtime to c++_shared.
        android_api_level(int): android_api_level
    """
    folder_name_parts = []
    if enable_asan:
        folder_name_parts.append("asan")
    if not enable_rtti:
        folder_name_parts.append("nortti")
    if not enable_exception:
        folder_name_parts.append("noexception")
    if enable_minimal_build:
        folder_name_parts.append("minimal")

    folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)

    file_name = f"{target_abi}-android.cmake"

    for config in configs:
        dest_path = Path(build_dir) / config / folder_name / file_name

        os.makedirs(dest_path.parent, exist_ok=True)

        with open(dest_path, "w", encoding="utf-8") as f:
            add_copyright_header(f)

            # Set target architecture for Android
            if target_abi == "arm-neon":
                f.write("set(VCPKG_TARGET_ARCHITECTURE arm)\n")
                f.write("set(VCPKG_MAKE_BUILD_TRIPLET --host=armv7a-linux-androideabi)\n")
                f.write(
                    "list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DANDROID_ABI=armeabi-v7a -DANDROID_ARM_NEON=ON -DCMAKE_ANDROID_ARM_NEON=ON)\n"
                )
            elif target_abi == "arm64":
                f.write("set(VCPKG_TARGET_ARCHITECTURE arm64)\n")
                f.write("set(VCPKG_MAKE_BUILD_TRIPLET --host=aarch64-linux-android)\n")
                f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DANDROID_ABI=arm64-v8a)\n")
            elif target_abi == "x64":
                f.write("set(VCPKG_TARGET_ARCHITECTURE x64)\n")
                f.write("set(VCPKG_MAKE_BUILD_TRIPLET --host=x86_64-linux-android)\n")
                f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DANDROID_ABI=x86_64)\n")
            elif target_abi == "x86":
                f.write("set(VCPKG_TARGET_ARCHITECTURE x86)\n")
                f.write("set(VCPKG_MAKE_BUILD_TRIPLET --host=i686-linux-android)\n")
                f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DANDROID_ABI=x86)\n")

            f.write(
                f"list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DANDROID_USE_LEGACY_TOOLCHAIN_FILE=false -DANDROID_PLATFORM=android-{android_api_level} -DANDROID_MIN_SDK={android_api_level})\n"
            )

            # Set CRT linkage
            # VCPKG_CRT_LINKAGE specifies the desired CRT linkage (for MSVC).
            # Valid options are dynamic and static.
            crt_linkage = "static"
            f.write(f"set(VCPKG_CRT_LINKAGE {crt_linkage})\n")

            # Set library linkage
            # VCPKG_LIBRARY_LINKAGE specifies the preferred library linkage.
            # Valid options are dynamic and static. Libraries can ignore this setting if they do not support the preferred linkage type. In our case, we prefer to use static libs.
            f.write("set(VCPKG_LIBRARY_LINKAGE static)\n")
            if not enable_rtti:
                f.write("set(CMAKE_ANDROID_RTTI OFF)\n")
            if not enable_exception:
                f.write("set(CMAKE_ANDROID_EXCEPTIONS OFF)\n")
            if use_cpp_shared:
                f.write("set(ANDROID_STL c++_shared)\n")

            ldflags = []

            cflags = ["-g", "-ffunction-sections", "-fdata-sections"]
            cflags_release = ["-DNDEBUG", "-O3"]

            if enable_asan:
                cflags += ["-fsanitize=address"]
                ldflags += ["-fsanitize=address"]

            ldflags.append("-g")

            cxxflags = cflags.copy()

            if not enable_rtti:
                cxxflags.append("-fno-rtti")

            if not enable_exception:
                cxxflags += ["-fno-exceptions", "-fno-unwind-tables", "-fno-asynchronous-unwind-tables"]

            if cflags:
                f.write(f'set(VCPKG_C_FLAGS "{" ".join(cflags)}")\n')

            if cxxflags:
                f.write(f'set(VCPKG_CXX_FLAGS "{" ".join(cxxflags)}")\n')

            if cflags_release:
                f.write(f'set(VCPKG_C_FLAGS_RELEASE "{" ".join(cflags_release)}")\n')
                f.write(f'set(VCPKG_CXX_FLAGS_RELEASE "{" ".join(cflags_release)}")\n')
                f.write(f'set(VCPKG_C_FLAGS_RELWITHDEBINFO "{" ".join(cflags_release)}")\n')
                f.write(f'set(VCPKG_CXX_FLAGS_RELWITHDEBINFO "{" ".join(cflags_release)}")\n')

            # Set target platform
            # VCPKG_CMAKE_SYSTEM_NAME specifies the target platform.
            f.write("set(VCPKG_CMAKE_SYSTEM_NAME Android)\n")
            f.write("set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n")
            f.write(
                "list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS --compile-no-warning-as-error -DBENCHMARK_ENABLE_WERROR=OFF)\n"
            )

            if ldflags:
                f.write(f'set(VCPKG_LINKER_FLAGS "{" ".join(ldflags)}")\n')
            f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_CXX_STANDARD=17)\n")
            add_build_type(f, config)
            add_port_configs(f, enable_exception, False, enable_minimal_build)  # Pass enable_minimal_build


def generate_android_triplets(
    build_dir: str, configs: Iterable[str], use_cpp_shared: bool, android_api_level: int
) -> None:
    """
    Generate triplet files for POSIX platforms (Linux, macOS, Android).

    Args:
        build_dir (str): The directory to save the generated triplet files.
    """
    target_abis = ["x64", "arm64", "arm-neon", "x86"]
    for enable_asan in [True, False]:
        for enable_rtti in [True, False]:
            for enable_exception in [True, False]:
                for enable_minimal_build in [True, False]:
                    if not enable_exception and not enable_minimal_build:
                        continue
                    for target_abi in target_abis:
                        generate_triplet_for_android(
                            build_dir,
                            configs,
                            target_abi,
                            enable_rtti,
                            enable_exception,
                            enable_asan,
                            enable_minimal_build,
                            use_cpp_shared,
                            android_api_level,
                        )


def generate_triplet_for_posix_platform(
    build_dir: str,
    configs: Iterable[str],
    os_name: str,
    enable_rtti: bool,
    enable_exception: bool,
    enable_binskim: bool,
    enable_asan: bool,
    enable_minimal_build: bool,
    crt_linkage: str,
    target_abi: str,
    osx_deployment_target: str,
) -> None:
    """
    Generate triplet file for POSIX platforms (Linux, macOS).

    Args:
        build_dir (str): The directory to save the generated triplet files.
        os_name (str): The name of the operating system.
        enable_rtti (bool): Flag indicating if RTTI is enabled.
        enable_exception (bool): Flag indicating if exceptions are enabled.
        enable_binskim (bool): Flag indicating if BinSkim is enabled.
        enable_asan (bool): Flag indicating if AddressSanitizer is enabled.
        enable_minimal_build (bool): Flag indicating if ONNX minimal build is enabled.
        crt_linkage (str): The CRT linkage type ("static" or "dynamic").
        target_abi (str): The target ABI, which maps to the VCPKG_TARGET_ARCHITECTURE variable. Valid options include x86, x64, arm, arm64, arm64ec, s390x, ppc64le, riscv32, riscv64, loongarch32, loongarch64, mips64.
        osx_deployment_target (str, optional): The macOS deployment target version. The parameter sets the minimum macOS version for compiled binaries. It also changes what versions of the macOS platform SDK CMake will search for. See the CMake documentation for CMAKE_OSX_DEPLOYMENT_TARGET for more information.
    """
    folder_name_parts = []
    if enable_asan:
        folder_name_parts.append("asan")
    if enable_binskim:
        folder_name_parts.append("binskim")
    if not enable_rtti:
        folder_name_parts.append("nortti")
    if not enable_exception:
        folder_name_parts.append("noexception")
    if enable_minimal_build:
        folder_name_parts.append("minimal")

    folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)

    file_name = f"{target_abi}-{os_name}.cmake"

    for config in configs:
        dest_path = Path(build_dir) / config / folder_name / file_name

        os.makedirs(dest_path.parent, exist_ok=True)

        with open(dest_path, "w", encoding="utf-8") as f:
            add_copyright_header(f)

            # Set target architecture based on `os_name` and `target_abi`.
            #
            # In most cases VCPKG itself can help automatically detect the target architecture, but sometimes it is not as what we want. The following code process the special cases.
            if target_abi == "universal2":
                # Assume the host machine is Intel based
                f.write("set(VCPKG_TARGET_ARCHITECTURE x64)\n")
            else:
                f.write(f"set(VCPKG_TARGET_ARCHITECTURE {target_abi})\n")

            # Set CRT linkage
            # VCPKG_CRT_LINKAGE specifies the desired CRT linkage (for MSVC).
            # Valid options are dynamic and static.
            f.write(f"set(VCPKG_CRT_LINKAGE {crt_linkage})\n")

            # Set library linkage
            # VCPKG_LIBRARY_LINKAGE specifies the preferred library linkage.
            # Valid options are dynamic and static. Libraries can ignore this setting if they do not support the preferred linkage type.
            f.write("set(VCPKG_LIBRARY_LINKAGE static)\n")

            ldflags = []

            if enable_binskim and os_name == "linux":
                # BinSkim rule 3005: Enable stack clash protection
                # This check ensures that stack clash protection is enabled. Each program running on a computer uses a special memory region called the stack.
                # This memory region is special because it grows automatically when the program needs more stack memory. But if it grows too much and gets too close to another memory region,
                # the program may confuse the stack with the other memory region. An attacker can exploit this confusion to overwrite the stack with the other memory region, or the other way around.
                # Use the compiler flags '-fstack-clash-protection' to enable this.
                # BinSkim rule BA3011: Enable BIND_NOW
                # This check ensures that some relocation data is marked as read-only after the executable is loaded, and moved below the '.data' section in memory.
                # This prevents them from being overwritten, which can redirect control flow. Use the compiler flags '-Wl,-z,now' to enable this.
                ldflags = ["-Wl,-Bsymbolic-functions", "-Wl,-z,relro", "-Wl,-z,now", "-Wl,-z,noexecstack"]

            cflags = ["-g", "-ffunction-sections", "-fdata-sections"]
            cflags_release = ["-DNDEBUG", "-O3"]

            if enable_binskim:
                cflags_release += ["-Wp,-D_FORTIFY_SOURCE=2", "-Wp,-D_GLIBCXX_ASSERTIONS", "-fstack-protector-strong"]
                if target_abi == "x64":
                    cflags_release += ["-fstack-clash-protection", "-fcf-protection"]

            elif enable_asan:
                cflags += ["-fsanitize=address"]
                ldflags += ["-fsanitize=address"]

            ldflags.append("-g")

            if not enable_rtti:
                cflags.append("-DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0")

            cxxflags = cflags.copy()
            if os_name == "osx":
                cxxflags += ["-fvisibility=hidden", "-fvisibility-inlines-hidden"]
            if not enable_rtti:
                cxxflags.append("-fno-rtti")

            if not enable_exception:
                cxxflags += ["-fno-exceptions", "-fno-unwind-tables", "-fno-asynchronous-unwind-tables"]

            if cflags:
                f.write(f'set(VCPKG_C_FLAGS "{" ".join(cflags)}")\n')

            if cxxflags:
                f.write(f'set(VCPKG_CXX_FLAGS "{" ".join(cxxflags)}")\n')

            if cflags_release:
                f.write(f'set(VCPKG_C_FLAGS_RELEASE "{" ".join(cflags_release)}")\n')
                f.write(f'set(VCPKG_CXX_FLAGS_RELEASE "{" ".join(cflags_release)}")\n')
                f.write(f'set(VCPKG_C_FLAGS_RELWITHDEBINFO "{" ".join(cflags_release)}")\n')
                f.write(f'set(VCPKG_CXX_FLAGS_RELWITHDEBINFO "{" ".join(cflags_release)}")\n')

            # Set target platform
            # VCPKG_CMAKE_SYSTEM_NAME specifies the target platform.
            if os_name == "linux":
                f.write("set(VCPKG_CMAKE_SYSTEM_NAME Linux)\n")
            else:
                f.write("set(VCPKG_CMAKE_SYSTEM_NAME Darwin)\n")
                osx_abi = None
                if target_abi == "x64":
                    osx_abi = "x86_64"
                elif target_abi == "universal2":
                    osx_abi = "x86_64;arm64"
                else:
                    osx_abi = target_abi
                f.write(f'set(VCPKG_OSX_ARCHITECTURES "{osx_abi}")\n')
                if osx_deployment_target:
                    f.write(f'set(VCPKG_OSX_DEPLOYMENT_TARGET "{osx_deployment_target}")\n')
            f.write("set(CMAKE_POSITION_INDEPENDENT_CODE ON)\n")
            f.write(
                "list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS --compile-no-warning-as-error -DBENCHMARK_ENABLE_WERROR=OFF)\n"
            )

            if ldflags:
                f.write(f'set(VCPKG_LINKER_FLAGS "{" ".join(ldflags)}")\n')
            if os_name == "osx":
                f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_CXX_STANDARD=20)\n")
            else:
                f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DCMAKE_CXX_STANDARD=17)\n")
            add_build_type(f, config)
            add_port_configs(f, enable_exception, False, enable_minimal_build)  # Pass enable_minimal_build


def generate_vcpkg_triplets_for_emscripten(
    build_dir: str,
    configs: Iterable[str],
    emscripten_root: str,
    # Parameters defining the specific build configuration
    enable_rtti: bool,
    enable_wasm_exception_catching: bool,  # Controls -sDISABLE_EXCEPTION_CATCHING=...
    enable_minimal_onnx_build: bool,  # Controls ONNX port setting AND C++ exceptions (-fno-exceptions)
    enable_asan: bool,
) -> None:
    """
    Generate triplet files for Emscripten (WASM) for wasm32 and wasm64.
    Places files in the 'default' subdirectory.
    Configures flags based on passed parameters.

    Derives C++ exception support based on the minimal build flag:
    - If enable_minimal_onnx_build=True, C++ exceptions are disabled (-fno-exceptions).
    - If enable_minimal_onnx_build=False, C++ exceptions are assumed enabled (-fexceptions).

    This supports three main effective EH scenarios depending on the combination of
    'enable_minimal_onnx_build' and 'enable_wasm_exception_catching':
    1. No EH (-fno-exceptions, -sDISABLE_EXCEPTION_CATCHING=1):
       Set enable_minimal_onnx_build=True, enable_wasm_exception_catching=False
    2. Full EH (-fexceptions, -sDISABLE_EXCEPTION_CATCHING=0):
       Set enable_minimal_onnx_build=False, enable_wasm_exception_catching=True
    3. Throw Only EH (-fexceptions, -sDISABLE_EXCEPTION_CATCHING=1):
       Set enable_minimal_onnx_build=False, enable_wasm_exception_catching=False

    Args:
        build_dir (str): The directory to save the generated triplet files.
        emscripten_root (str): The root path of Emscripten.
        enable_rtti (bool): Flag indicating if RTTI is enabled for dependencies.
        enable_wasm_exception_catching (bool): Flag indicating if the Emscripten runtime
                                             exception catching mechanism should be enabled
                                             (controls -sDISABLE_EXCEPTION_CATCHING=...).
        enable_minimal_onnx_build (bool): Flag controlling if the ONNX dependency
                                        should be built with DONNX_MINIMAL_BUILD=ON.
                                        Also implicitly controls C++ exceptions for
                                        dependencies (True => -fno-exceptions).
        enable_asan (bool): Flag indicating if AddressSanitizer is enabled for dependencies.
    """
    # Always place generated files in the 'default' folder for Emscripten
    folder_name = "default"

    # Derive C++ exception enablement from the minimal build flag
    cpp_exceptions_enabled = not enable_minimal_onnx_build

    for target_abi in ["wasm32", "wasm64"]:
        os_name = "emscripten"
        file_name = f"{target_abi}-{os_name}.cmake"
        for config in configs:
            dest_path = Path(build_dir) / config / folder_name / file_name
            os.makedirs(dest_path.parent, exist_ok=True)

            with open(dest_path, "w", encoding="utf-8") as f:
                add_copyright_header(f)
                f.write(r"""
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Emscripten)
""")
                f.write(f"set(VCPKG_TARGET_ARCHITECTURE {target_abi})\n")
                emscripten_root_path_cmake_path = emscripten_root.replace("\\", "/")
                f.write(f'set(EMSCRIPTEN_ROOT_PATH "{emscripten_root_path_cmake_path}")\n')

                # Define the path to the intermediate toolchain file used by vcpkg for wasm
                vcpkg_toolchain_file = (Path(build_dir) / "emsdk_vcpkg_toolchain.cmake").absolute()
                vcpkg_toolchain_file_cmake_path = str(vcpkg_toolchain_file).replace("\\", "/")
                f.write(f'set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "{vcpkg_toolchain_file_cmake_path}")\n')

                # --- Configure Flags based on Parameters ---
                cflags_release = ["-DNDEBUG", "-O3", "-flto"]
                ldflags = []  # Initialize linker flags list
                # Base flags applicable to both C and C++
                base_flags = [
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-msimd128",
                    "-pthread",
                    "-Wno-pthreads-mem-growth",
                ]

                # ASan (apply to Base, Linker)
                if enable_asan:
                    asan_flag = "-fsanitize=address"
                    base_flags.append(asan_flag)
                    ldflags.append(asan_flag)  # Add to linker flags

                # Wasm Exception Catching Runtime (-s flag, apply to Base and Linker flags)
                exception_catching_flag = ""
                if enable_wasm_exception_catching:
                    exception_catching_flag = "-sDISABLE_EXCEPTION_CATCHING=0"
                else:
                    exception_catching_flag = "-sDISABLE_EXCEPTION_CATCHING=1"

                base_flags.append(exception_catching_flag)  # Add to base C/C++ flags
                ldflags.append(exception_catching_flag)  # Add to linker flags

                # Wasm64 Memory (apply to Base, Linker)
                if target_abi == "wasm64":
                    memory_flag = "-sMEMORY64"
                    base_flags.append(memory_flag)
                    ldflags.append(memory_flag)  # Add to linker flags

                # --- C Flags ---
                # VCPKG_C_FLAGS applies only base flags
                f.write(f'set(VCPKG_C_FLAGS "{" ".join(base_flags)}")\n')

                # --- CXX Flags ---
                # Start with base flags
                cxxflags = list(base_flags)  # Create a copy

                # C++ RTTI Compiler Flag
                if not enable_rtti:
                    cxxflags.append("-fno-rtti")

                # C++ Exceptions Compiler Flag (Derived from enable_minimal_onnx_build)
                if not cpp_exceptions_enabled:  # i.e., if enable_minimal_onnx_build is True
                    cxxflags.append("-fno-exceptions")
                # If cpp_exceptions_enabled=True, we assume -fexceptions is the default
                # or handled by the Emscripten toolchain/CMake settings elsewhere.

                f.write(f'set(VCPKG_CXX_FLAGS "{" ".join(cxxflags)}")\n')

                # --- Linker Flags ---
                # Apply Linker flags (now includes exception and memory flags explicitly)
                if len(ldflags) >= 1:
                    f.write('set(VCPKG_LINKER_FLAGS "{}")\n'.format(" ".join(ldflags)))

                # --- Release / RelWithDebInfo Flags ---
                # Combine base flags with release-specific flags
                c_combined_release_flags = cflags_release + base_flags
                cxx_combined_release_flags = cflags_release + cxxflags  # Use the derived cxxflags

                f.write(f'set(VCPKG_C_FLAGS_RELEASE "{" ".join(c_combined_release_flags)}")\n')
                f.write(f'set(VCPKG_CXX_FLAGS_RELEASE "{" ".join(cxx_combined_release_flags)}")\n')

                f.write("set(VCPKG_LINKER_FLAGS_RELEASE -flto)\n")

                add_build_type(f, config)

                # --- Add Port Specific Configs ---
                # Pass the derived C++ exception status and the original minimal build flag
                add_port_configs(
                    f,
                    has_exception=cpp_exceptions_enabled,  # Derived value
                    is_emscripten=True,
                    enable_minimal_build=enable_minimal_onnx_build,
                )  # Original parameter


def generate_windows_triplets(build_dir: str, configs: Iterable[str], toolset_version: str) -> None:
    """
    Generate triplet files for Windows platforms.

    Args:
        build_dir (str): The directory to save the generated triplet files.
        toolset_version (str, optional): The version of the platform toolset.
    """
    # Below are all the CPU ARCHs we support on Windows.
    # ARM64 is for ARM64 processes that contains traditional ARM64 code.
    # ARM64EC is a different ABI that utilizes a subset of the ARM64 register set to provide interoperability with x64 code.
    # Both ARM64 and ARM64EC for are AArch64 CPUs.
    # We have dropped the support for ARM32.
    target_abis = ["x86", "x64", "arm64", "arm64ec"]
    crt_linkages = ["static", "dynamic"]
    for enable_rtti in [True, False]:
        for enable_exception in [True, False]:
            for enable_binskim in [True, False]:
                for enable_asan in [True, False]:
                    for enable_minimal_build in [True, False]:
                        for crt_linkage in crt_linkages:
                            # Address Sanitizer libs do not have a Qspectre version. So they two cannot be both enabled.
                            if enable_asan and enable_binskim:
                                continue
                            # ORT Constraint: If exceptions are disabled, minimal build must be enabled
                            if not enable_exception and not enable_minimal_build:
                                continue

                            for target_abi in target_abis:
                                folder_name_parts = []
                                if enable_asan:
                                    folder_name_parts.append("asan")
                                if enable_binskim:
                                    folder_name_parts.append("binskim")
                                if not enable_rtti:
                                    folder_name_parts.append("nortti")
                                if not enable_exception:
                                    folder_name_parts.append("noexception")
                                if enable_minimal_build:
                                    folder_name_parts.append("minimal")

                                folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)
                                file_name_parts = [target_abi, "windows", "static"]
                                if crt_linkage == "dynamic":
                                    file_name_parts.append("md")
                                file_name = "-".join(file_name_parts) + ".cmake"
                                for config in configs:
                                    dest_path = Path(build_dir) / config / folder_name / file_name
                                    os.makedirs(dest_path.parent, exist_ok=True)
                                    with open(dest_path, "w", encoding="utf-8") as f:
                                        add_copyright_header(f)
                                        f.write(f"set(VCPKG_TARGET_ARCHITECTURE {target_abi})\n")
                                        f.write(f"set(VCPKG_CRT_LINKAGE {crt_linkage})\n")
                                        f.write("set(VCPKG_LIBRARY_LINKAGE static)\n")
                                        if toolset_version:
                                            f.write(f"set(VCPKG_PLATFORM_TOOLSET_VERSION {toolset_version})\n")
                                        cflags = ["/MP", "/DWIN32", "/D_WINDOWS"]
                                        if enable_binskim:
                                            cflags += [
                                                "/DWINAPI_FAMILY=100",
                                                "/DWINVER=0x0A00",
                                                "/D_WIN32_WINNT=0x0A00",
                                                "/DNTDDI_VERSION=0x0A000000",
                                            ]
                                        ldflags = []
                                        if enable_binskim:
                                            cflags += ["/guard:cf", "/Qspectre", "/W3"]
                                            ldflags = ["/profile", "/DYNAMICBASE"]
                                        elif enable_asan:
                                            cflags.append("/fsanitize=address")
                                        cxxflags = cflags.copy()
                                        cxxflags.append("/Zc:__cplusplus")
                                        if enable_exception:
                                            cxxflags.append("/EHsc")
                                        # MSVC doesn't have a specific flag to disable exceptions like /EHs-c-
                                        # but relies on _HAS_EXCEPTIONS=0 and potentially other flags managed by ORT's main CMake.
                                        # Vcpkg doesn't directly control this via a simple triplet flag AFAIK.
                                        # ORT's CMake handles this via CMAKE_CXX_FLAGS adjustment.
                                        if not enable_rtti:
                                            cxxflags += ["/GR-", "/we4541"]
                                        if cflags:
                                            f.write(f'set(VCPKG_C_FLAGS "{" ".join(cflags)}")\n')
                                        if cxxflags:
                                            f.write(f'set(VCPKG_CXX_FLAGS "{" ".join(cxxflags)}")\n')
                                        f.write(
                                            "list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS --compile-no-warning-as-error -DCMAKE_CXX_STANDARD=17)\n"
                                        )
                                        if ldflags:
                                            f.write(f'set(VCPKG_LINKER_FLAGS "{" ".join(ldflags)}")\n')
                                        add_build_type(f, config)
                                        add_port_configs(
                                            f, enable_exception, False, enable_minimal_build
                                        )  # Pass enable_minimal_build


def generate_linux_triplets(build_dir: str, configs: Iterable[str]) -> None:
    """
    Generate triplet files for Linux platforms.

    Args:
        build_dir (str): The directory to save the generated triplet files.
    """
    target_abis = ["x86", "x64", "arm", "arm64", "s390x", "ppc64le", "riscv64", "loongarch64", "mips64"]
    for enable_rtti in [True, False]:
        for enable_exception in [True, False]:
            for enable_binskim in [True, False]:
                for enable_asan in [True, False]:
                    for enable_minimal_build in [True, False]:
                        if enable_asan and enable_binskim:
                            continue
                        if not enable_exception and not enable_minimal_build:
                            continue
                        for target_abi in target_abis:
                            generate_triplet_for_posix_platform(
                                build_dir,
                                configs,
                                "linux",
                                enable_rtti,
                                enable_exception,
                                enable_binskim,
                                enable_asan,
                                enable_minimal_build,
                                "dynamic",
                                target_abi,
                                None,
                            )


def generate_macos_triplets(build_dir: str, configs: Iterable[str], osx_deployment_target: str) -> None:
    """
    Generate triplet files for macOS platforms.

    Args:
        build_dir (str): The directory to save the generated triplet files.
        osx_deployment_target (str, optional): The macOS deployment target version.
    """
    target_abis = ["x64", "arm64", "universal2"]
    for enable_rtti in [True, False]:
        for enable_exception in [True, False]:
            for enable_binskim in [True, False]:
                for enable_asan in [True, False]:
                    for enable_minimal_build in [True, False]:
                        if enable_asan and enable_binskim:
                            continue
                        # ORT Constraint: If exceptions are disabled, minimal build must be enabled
                        if not enable_exception and not enable_minimal_build:
                            continue
                        for target_abi in target_abis:
                            generate_triplet_for_posix_platform(
                                build_dir,
                                configs,
                                "osx",
                                enable_rtti,
                                enable_exception,
                                enable_binskim,
                                enable_asan,
                                enable_minimal_build,
                                "dynamic",
                                target_abi,
                                osx_deployment_target,
                            )
