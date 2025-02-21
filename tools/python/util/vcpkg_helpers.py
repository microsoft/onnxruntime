import os
from pathlib import Path

# The official vcpkg repository has about 80 different triplets. But ONNX Runtime has many more build variants. For example, in general, for each platform, we need to support builds with C++ exceptions, builds without C++ exceptions, builds with C++ RTTI, builds without C++ RTTI, linking to static C++ runtime, linking to dynamic (shared) C++ runtime, builds with address sanitizer, builds without address sanitizer, etc. Therefore, this script file was created to dynamically generate the triplet files on-the-fly.

# Originally, we tried to check in all the generated files into our repository so that people could build onnxruntime without using build.py or any other Python scripts in the "/tools" directory. However, we encountered an issue when adding support for WASM builds. VCPKG has a limitation that when doing cross-compiling, the triplet file must specify the full path of the chain-loaded toolchain file. The file needs to be located either via environment variables (like ANDROID_NDK_HOME) or via an absolute path. Since environment variables are hard to track, we chose the latter approach. So the generated triplet files may contain absolute file paths that are only valid on the current build machine.

# The compiler flags(CFLAGS/CXXFLAGS/LDFLAGS) settings in this file must be consistent with the cmake code in "cmake/adjust_global_compile_flags.cmake" so that all the statically linked code were compiled by the same set of compile flags.


# This is a way to add customizations to the official VCPKG ports.
def add_port_configs(f, has_exception: bool, is_emscripten: bool) -> None:
    """
    Add port-specific configurations to the triplet file.

    Args:
        f (file object): The file object to write configurations.
        has_exception (bool): Flag indicating if exceptions are enabled.
    """
    f.write(
        r"""if(PORT MATCHES "onnx")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_DISABLE_STATIC_REGISTRATION=ON"
    )
endif()
if(PORT MATCHES "benchmark")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DBENCHMARK_ENABLE_WERROR=OFF"
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
    if not has_exception:
        f.write(
            r"""if(PORT MATCHES "onnx")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
        "-DONNX_DISABLE_EXCEPTIONS=ON"
    )
endif()
"""
        )


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


def generate_triplet_for_android(
    build_dir: str,
    target_abi: str,
    enable_rtti: bool,
    enable_exception: bool,
    enable_asan: bool,
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

    folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)

    file_name = f"{target_abi}-android.cmake"

    dest_path = Path(build_dir) / folder_name / file_name

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
            f.write("set(CMAKE_ANDROID_RTTI OFF)")
        if not enable_exception:
            f.write("set(CMAKE_ANDROID_EXCEPTIONS OFF)")
        if use_cpp_shared:
            f.write("set(ANDROID_STL c++_shared)")

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
        add_port_configs(f, enable_exception, False)


def generate_android_triplets(build_dir: str, use_cpp_shared: bool, android_api_level: int) -> None:
    """
    Generate triplet files for POSIX platforms (Linux, macOS, Android).

    Args:
        build_dir (str): The directory to save the generated triplet files.
    """
    target_abis = ["x64", "arm64", "arm-neon", "x86"]
    for enable_asan in [True, False]:
        for enable_rtti in [True, False]:
            for enable_exception in [True, False]:
                for target_abi in target_abis:
                    generate_triplet_for_android(
                        build_dir,
                        target_abi,
                        enable_rtti,
                        enable_exception,
                        enable_asan,
                        use_cpp_shared,
                        android_api_level,
                    )


def generate_triplet_for_posix_platform(
    build_dir: str,
    os_name: str,
    enable_rtti: bool,
    enable_exception: bool,
    enable_binskim: bool,
    enable_asan: bool,
    crt_linkage: str,
    target_abi: str,
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
        crt_linkage (str): The CRT linkage type ("static" or "dynamic").
        target_abi (str): The target ABI, which maps to the VCPKG_TARGET_ARCHITECTURE variable. Valid options include x86, x64, arm, arm64, arm64ec, s390x, ppc64le, riscv32, riscv64, loongarch32, loongarch64, mips64.
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

    folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)

    file_name = f"{target_abi}-{os_name}.cmake"

    dest_path = Path(build_dir) / folder_name / file_name

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
        add_port_configs(f, enable_exception, False)


def generate_vcpkg_triplets_for_emscripten(build_dir: str, emscripten_root: str) -> None:
    """
    Generate triplet files for Emscripten (WASM).

    Args:
        build_dir (str): The directory to save the generated triplet files.
        emscripten_root (str): The root path of Emscripten.
    """
    for enable_rtti in [True, False]:
        for enable_asan in [True, False]:
            for target_abi in ["wasm32", "wasm64"]:
                folder_name_parts = []
                if enable_asan:
                    folder_name_parts.append("asan")
                folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)
                os_name = "emscripten"
                folder_name_parts = []
                if enable_asan:
                    folder_name_parts.append("asan")
                if not enable_rtti:
                    folder_name_parts.append("nortti")
                folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)
                file_name = f"{target_abi}-{os_name}.cmake"
                dest_path = Path(build_dir) / folder_name / file_name
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
                    vcpkg_toolchain_file = (Path(build_dir) / "emsdk_vcpkg_toolchain.cmake").absolute()
                    vcpkg_toolchain_file_cmake_path = str(vcpkg_toolchain_file).replace("\\", "/")
                    f.write(f'set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "{vcpkg_toolchain_file_cmake_path}")\n')
                    cflags_release = ["-DNDEBUG", "-O3", "-pthread"]
                    ldflags = []
                    cflags = [
                        "-ffunction-sections",
                        "-fdata-sections",
                        "-msimd128",
                        "-pthread",
                        "-Wno-pthreads-mem-growth",
                        "-sDISABLE_EXCEPTION_CATCHING=0",
                    ]
                    if enable_asan:
                        cflags += ["-fsanitize=address"]
                        ldflags += ["-fsanitize=address"]
                    if target_abi == "wasm64":
                        cflags.append("-sMEMORY64")
                        ldflags.append("-sMEMORY64")
                    if len(ldflags) >= 1:
                        f.write('set(VCPKG_LINKER_FLAGS "{}")\n'.format(" ".join(ldflags)))
                    cxxflags = cflags.copy()
                    if cflags:
                        f.write(f'set(VCPKG_C_FLAGS "{" ".join(cflags)}")\n')
                    if cxxflags:
                        f.write(f'set(VCPKG_CXX_FLAGS "{" ".join(cxxflags)}")\n')
                    if cflags_release:
                        cflags_release += cflags
                        f.write(f'set(VCPKG_C_FLAGS_RELEASE "{" ".join(cflags_release)}")\n')
                        f.write(f'set(VCPKG_CXX_FLAGS_RELEASE "{" ".join(cflags_release)}")\n')
                        f.write(f'set(VCPKG_C_FLAGS_RELWITHDEBINFO "{" ".join(cflags_release)}")\n')
                        f.write(f'set(VCPKG_CXX_FLAGS_RELWITHDEBINFO "{" ".join(cflags_release)}")\n')
                    add_port_configs(f, True, True)


def generate_windows_triplets(build_dir: str) -> None:
    """
    Generate triplet files for Windows platforms.

    Args:
        build_dir (str): The directory to save the generated triplet files.
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
                    for crt_linkage in crt_linkages:
                        # Address Sanitizer libs do not have a Qspectre version. So they two cannot be both enabled.
                        if enable_asan and enable_binskim:
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
                            folder_name = "default" if len(folder_name_parts) == 0 else "_".join(folder_name_parts)
                            file_name_parts = [target_abi, "windows", "static"]
                            if crt_linkage == "dynamic":
                                file_name_parts.append("md")
                            file_name = "-".join(file_name_parts) + ".cmake"
                            dest_path = Path(build_dir) / folder_name / file_name
                            os.makedirs(dest_path.parent, exist_ok=True)
                            with open(dest_path, "w", encoding="utf-8") as f:
                                add_copyright_header(f)
                                f.write(f"set(VCPKG_TARGET_ARCHITECTURE {target_abi})\n")
                                f.write(f"set(VCPKG_CRT_LINKAGE {crt_linkage})\n")
                                f.write("set(VCPKG_LIBRARY_LINKAGE static)\n")
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
                                add_port_configs(f, enable_exception, False)


def generate_posix_triplets(build_dir: str) -> None:
    """
    Generate triplet files for POSIX platforms (Linux, macOS).

    Args:
        build_dir (str): The directory to save the generated triplet files.
    """
    for os_name in ["linux", "osx"]:
        if os_name == "linux":
            target_abis = ["x86", "x64", "arm", "arm64", "s390x", "ppc64le", "riscv64", "loongarch64", "mips64"]
        else:
            target_abis = ["x64", "arm64", "universal2"]
        for enable_rtti in [True, False]:
            for enable_exception in [True, False]:
                for enable_binskim in [True, False]:
                    for enable_asan in [True, False]:
                        if enable_asan and enable_binskim:
                            continue
                        for target_abi in target_abis:
                            generate_triplet_for_posix_platform(
                                build_dir,
                                os_name,
                                enable_rtti,
                                enable_exception,
                                enable_binskim,
                                enable_asan,
                                "dynamic",
                                target_abi,
                            )
