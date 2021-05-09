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
import sys


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


def validate_build_package_info():
    try:
        from onnxruntime.training.ortmodule import ORTModule # noqa
        has_ortmodule = True
    except ImportError:
        has_ortmodule = False
        print("no ortmodule")
    except: # noqa
        # this may happen if Cuda is not installed
        has_ortmodule = True

    package_name = ''
    version = ''
    cuda_version = ''

    if has_ortmodule:
        try:
            # collect onnxruntime package name, version, and cuda version
            from .build_and_package_info import package_name
            from .build_and_package_info import __version__ as version

            cuda_version = None
            try:
                from .build_and_package_info import cuda_version
            except: # noqa
                pass

            print('onnxruntime training package info: package_name:', package_name, file=sys.stderr)
            print('onnxruntime training package info: __version__:', version, file=sys.stderr)

            if cuda_version:
                print('onnxruntime training package info: cuda_version:', cuda_version, file=sys.stderr)

                # collect cuda library build info. the library info may not be available
                # when the build environment has none or multiple libraries installed
                try:
                    from .build_and_package_info import cudart_version
                    print('onnxruntime build info: cudart_version:', cudart_version, file=sys.stderr)
                except: # noqa
                    print('WARNING: failed to get cudart_version from onnxruntime build info.', file=sys.stderr)
                    cudart_version = None

                try:
                    from .build_and_package_info import cudnn_version
                    print('onnxruntime build info: cudnn_version:', cudnn_version, file=sys.stderr)
                except: # noqa
                    print('WARNING: failed to get cudnn_version from onnxruntime build info', file=sys.stderr)
                    cudnn_version = None

                # collection cuda library info from current environment.
                from onnxruntime.capi.onnxruntime_collect_build_info import find_cudart_versions, find_cudnn_versions
                local_cudart_versions = find_cudart_versions(build_env=False)
                if cudart_version and cudart_version not in local_cudart_versions:
                    print('WARNING: failed to find cudart version that matches onnxruntime build info', file=sys.stderr)
                    print('WARNING: found cudart versions: ', local_cudart_versions, file=sys.stderr)

                local_cudnn_versions = find_cudnn_versions(build_env=False)
                if cudnn_version and cudnn_version not in local_cudnn_versions:
                    # need to be soft on cudnn version
                    # very likely there is a mismatch but onnxruntime works just fine.
                    print('INFO: failed to find cudnn version that matches onnxruntime build info', file=sys.stderr)
                    print('INFO: found cudnn versions: ', local_cudnn_versions, file=sys.stderr)
            else:
                # TODO: rcom
                pass

        except: # noqa
            print('WARNING: failed to collect onnxruntime version and build info', file=sys.stderr)

    return has_ortmodule, package_name, version, cuda_version


has_ortmodule, package_name, version, cuda_version = validate_build_package_info()
