# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Check OS requirements for ONNX Runtime Python Bindings.
"""
import platform
import linecache
import warnings


def check_distro_info():
    __my_distro__ = ''
    __my_distro_ver__ = ''
    __my_system__ = platform.system().lower()

    __OS_RELEASE_FILE__ = '/etc/os-release'
    __LSB_RELEASE_FILE__ = '/etc/lsb-release'

    if __my_system__ == 'windows':
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if __my_distro_ver__ != '10':
            warnings.warn('Unsupported Windows version (%s). ONNX Runtime supports Windows 10 and above, only.' %
                          __my_distro_ver__)
    elif __my_system__ == 'linux':
        ''' Although the 'platform' python module for getting Distro information works well on standard OS images
        running on real hardware, it is not accurate when running on Azure VMs, Git Bash, Cygwin, etc.
        The returned values for release and version are unpredictable for virtualized or emulated environments.
        /etc/os-release and /etc/lsb_release files, on the other hand, are guaranteed to exist and have standard values
        in all OSes supported by onnxruntime. The former is the current standard file to check OS info and the latter
        is its predecessor.
        '''
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
    elif __my_system__ == 'darwin':
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if int(__my_distro_ver__.split('.')[0]) < 11:
            warnings.warn('Unsupported macOS version (%s). ONNX Runtime supports macOS 11.0 or later.' %
                          (__my_distro_ver__))
    else:
        warnings.warn('Unsupported platform (%s). ONNX Runtime supports Linux, macOS and Windows platforms, only.' %
                      __my_system__)
