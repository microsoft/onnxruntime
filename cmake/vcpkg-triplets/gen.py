#!/usr/bin/python3

import os


# provide build options for vcpkg installed packages
def add_port_configs(f):
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


# provide build options for vcpkg installed packages
def add_copyright_header(f):
    f.write(
        r"""# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
    )


# Gen Windows triplets
target_abis = ["x86", "x64", "arm64", "arm64ec"]
crt_linkages = ["static", "dynamic"]
for enable_rtti in [True, False]:
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
                    if len(folder_name_parts) == 0:
                        folder_name = "default"
                    else:
                        folder_name = "_".join(folder_name_parts)
                    file_name_parts = [target_abi, "windows", "static"]
                    if crt_linkage == "dynamic":
                        file_name_parts.append("md")
                    file_name = "-".join(file_name_parts) + ".cmake"
                    dest_path = os.path.join(folder_name, file_name)
                    print(f"Creating file {dest_path}")
                    os.makedirs(folder_name, exist_ok=True)
                    with open(dest_path, "w", encoding="utf-8") as f:
                        add_copyright_header(f)
                        f.write(f"set(VCPKG_TARGET_ARCHITECTURE {target_abi})\n")
                        f.write(f"set(VCPKG_CRT_LINKAGE {crt_linkage})\n")
                        # we only support static. The default triplets use dynamic.
                        # The default triplets do not work with asan(protoc.exe crashes at startup). Therefore we do not override the default triplets and do not append asan to them.
                        # The default triplets are used for generating host dependencies(such as protoc.exe)
                        f.write("set(VCPKG_LIBRARY_LINKAGE static)\n")
                        cflags = ["/MP", "/DWIN32", "/D_WINDOWS"]
                        # TODO: if not gdk
                        cflags += [
                            "/DWINAPI_FAMILY=100",
                            "/DWINVER=0x0A00",
                            "/D_WIN32_WINNT=0x0A00",
                            "/DNTDDI_VERSION=0x0A000000",
                        ]
                        cxxflags = None
                        ldflags = None
                        cflags += []
                        if enable_binskim:
                            cflags += ["/guard:cf", "/Qspectre"]
                            ldflags = ["/profile", "/DYNAMICBASE"]
                        elif enable_asan:
                            cflags.append("/fsanitize=address")
                        cxxflags = cflags.copy()
                        cxxflags.append("/Zc:__cplusplus")
                        if not enable_rtti:
                            # Disable RTTI and turn usage of dynamic_cast and typeid into errors
                            cxxflags += ["/GR-", "/we4541"]
                        # TODO: should it be a cmake list separated by semicolons?
                        f.write('set(VCPKG_C_FLAGS "{}")\n'.format(" ".join(cflags)))
                        f.write('set(VCPKG_CXX_FLAGS "{}")\n'.format(" ".join(cxxflags)))
                        f.write("list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS --compile-no-warning-as-error)\n")
                        if ldflags:
                            f.write('set(VCPKG_LINKER_FLAGS "{}")\n'.format(" ".join(ldflags)))
                        add_port_configs(f)


# Gen Linux triplets
crt_linkages = ["static", "dynamic"]
for os_name in ["linux", "osx"]:
    if os_name == "linux":
        target_abis = ["x64", "arm64"]
    else:
        target_abis = ["x64", "arm64", "universal2"]
    for enable_rtti in [True, False]:
        for enable_binskim in [True, False]:
            for enable_asan in [True, False]:
                for crt_linkage in crt_linkages:
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
                        if len(folder_name_parts) == 0:
                            folder_name = "default"
                        else:
                            folder_name = "_".join(folder_name_parts)
                        file_name_parts = [target_abi, os_name]
                        file_name = "-".join(file_name_parts) + ".cmake"
                        dest_path = os.path.join(folder_name, file_name)
                        print(f"Creating file {dest_path}")
                        os.makedirs(folder_name, exist_ok=True)
                        with open(dest_path, "w", encoding="utf-8") as f:
                            add_copyright_header(f)
                            if target_abi == "universal2":
                                # Assume the host machine is Intel based
                                f.write("set(VCPKG_TARGET_ARCHITECTURE x64)\n")
                            else:
                                f.write(f"set(VCPKG_TARGET_ARCHITECTURE {target_abi})\n")
                            f.write(f"set(VCPKG_CRT_LINKAGE {crt_linkage})\n")
                            f.write("set(VCPKG_LIBRARY_LINKAGE static)\n")
                            if enable_binskim and os_name == "linux":
                                ldflags = [
                                    "-Wl,-Bsymbolic-functions",
                                    "-Wl,-z,relro",
                                    "-Wl,-z,now",
                                    "-Wl,-z,noexecstack",
                                ]
                            else:
                                ldflags = []
                            cflags = []
                            if enable_binskim:
                                cflags += [
                                    "-Wp,-D_FORTIFY_SOURCE=2",
                                    "-Wp,-D_GLIBCXX_ASSERTIONS",
                                    "-fstack-protector-strong",
                                ]
                                if target_abi == "x64":
                                    cflags += ["-fstack-clash-protection", "-fcf-protection"]
                            elif enable_asan:
                                cflags += ["-fsanitize=address"]
                                ldflags += ["-fsanitize=address"]
                            # Avoid unboundTypeError for WebNN EP since unbound type names are illegal with RTTI disabled
                            # in Embind API, relevant issue: https://github.com/emscripten-core/emscripten/issues/7001
                            if not enable_rtti:
                                cflags.append("-DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0")
                            cxxflags = cflags.copy()
                            if not enable_rtti:
                                cxxflags.append("-fno-rtti")
                            f.write('set(VCPKG_C_FLAGS "{}")\n'.format(" ".join(cflags)))
                            f.write('set(VCPKG_CXX_FLAGS "{}")\n'.format(" ".join(cxxflags)))
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
                                f.write('set(VCPKG_LINKER_FLAGS "{}")\n'.format(" ".join(ldflags)))
                            add_port_configs(f)
