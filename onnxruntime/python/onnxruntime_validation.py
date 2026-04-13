# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Check OS requirements for ONNX Runtime Python Bindings.
"""

import linecache
import platform
import warnings


def check_distro_info():
    __my_distro__ = ""
    __my_distro_ver__ = ""
    __my_system__ = platform.system().lower()

    __OS_RELEASE_FILE__ = "/etc/os-release"  # noqa: N806
    __LSB_RELEASE_FILE__ = "/etc/lsb-release"  # noqa: N806

    if __my_system__ == "windows":
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if __my_distro_ver__ not in ["10", "11", "2016server", "2019server", "2022server", "2025server"]:
            warnings.warn(
                f"Unsupported Windows version ({__my_distro_ver__}). ONNX Runtime supports Windows 10 and above, or Windows Server 2016 and above."
            )
    elif __my_system__ == "linux":
        """Although the 'platform' python module for getting Distro information works well on standard OS images
        running on real hardware, it is not accurate when running on Azure VMs, Git Bash, Cygwin, etc.
        The returned values for release and version are unpredictable for virtualized or emulated environments.
        /etc/os-release and /etc/lsb_release files, on the other hand, are guaranteed to exist and have standard values
        in all OSes supported by onnxruntime. The former is the current standard file to check OS info and the latter
        is its predecessor.
        """
        # Newer systems have /etc/os-release with relevant distro info
        __my_distro__ = linecache.getline(__OS_RELEASE_FILE__, 3)[3:-1]
        __my_distro_ver__ = linecache.getline(__OS_RELEASE_FILE__, 6)[12:-2]

        # Older systems may have /etc/os-release instead
        if not __my_distro__:
            __my_distro__ = linecache.getline(__LSB_RELEASE_FILE__, 1)[11:-1]
            __my_distro_ver__ = linecache.getline(__LSB_RELEASE_FILE__, 2)[16:-1]

        # Instead of trying to parse distro specific files,
        # warn the user ONNX Runtime may not work out of the box
        __my_distro__ = __my_distro__.lower()
        __my_distro_ver__ = __my_distro_ver__.lower()
    elif __my_system__ == "darwin":
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if int(__my_distro_ver__.split(".")[0]) < 11:
            warnings.warn(
                f"Unsupported macOS version ({__my_distro_ver__}). ONNX Runtime supports macOS 11.0 or later."
            )
    elif __my_system__ == "aix":
        import subprocess  # noqa: PLC0415

        returned_output = subprocess.check_output("oslevel")
        __my_distro_ver__str = returned_output.decode("utf-8")
        __my_distro_ver = __my_distro_ver__str[:3]
    else:
        warnings.warn(
            f"Unsupported platform ({__my_system__}). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only."
        )


def get_package_name_and_version_info():
    package_name = ""
    version = ""
    cuda_version = ""

    try:
        from .build_and_package_info import __version__ as version  # noqa: PLC0415
        from .build_and_package_info import package_name  # noqa: PLC0415

        try:  # noqa: SIM105
            from .build_and_package_info import cuda_version  # noqa: PLC0415
        except ImportError:
            # cuda_version is optional. For example, cpu only package does not have the attribute.
            pass
    except Exception as e:
        warnings.warn("WARNING: failed to collect package name and version info")
        print(e)

    return package_name, version, cuda_version



