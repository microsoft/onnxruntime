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


def validate_build_package_info():
    import_ortmodule_exception = None
    try:
        from onnxruntime.training.ortmodule import ORTModule # noqa
        has_ortmodule = True
    except ImportError:
        has_ortmodule = False
    except Exception as e:
        # this may happen if Cuda is not installed, we want to raise it after
        # for any exception other than not having ortmodule, we want to continue
        # device version validation and raise the exception after.
        import_ortmodule_exception = e
        has_ortmodule = True

    package_name = ''
    version = ''
    cuda_version = ''

    if has_ortmodule:
        try:
            # collect onnxruntime package name, version, and cuda version
            from .build_and_package_info import package_name
            from .build_and_package_info import __version__ as version

            try:
                from .build_and_package_info import cuda_version
            except: # noqa
                pass

            if cuda_version:
                # collect cuda library build info. the library info may not be available
                # when the build environment has none or multiple libraries installed
                try:
                    from .build_and_package_info import cudart_version
                except: # noqa
                    warnings.warn('WARNING: failed to get cudart_version from onnxruntime build info.')
                    cudart_version = None

                def print_build_package_info():
                    warnings.warn('onnxruntime training package info: package_name: %s' % package_name)
                    warnings.warn('onnxruntime training package info: __version__: %s' % version)
                    warnings.warn('onnxruntime training package info: cuda_version: %s' % cuda_version)
                    warnings.warn('onnxruntime build info: cudart_version: %s' % cudart_version)

                # collection cuda library info from current environment.
                from onnxruntime.capi.onnxruntime_collect_build_info import find_cudart_versions
                local_cudart_versions = find_cudart_versions(build_env=False, build_cuda_version=cuda_version)
                if cudart_version and cudart_version not in local_cudart_versions:
                    print_build_package_info()
                    warnings.warn('WARNING: failed to find cudart version that matches onnxruntime build info')
                    warnings.warn('WARNING: found cudart versions: %s' % local_cudart_versions)
            else:
                # TODO: rcom
                pass

        except Exception as e: # noqa
            warnings.warn('WARNING: failed to collect onnxruntime version and build info')
            print(e)

    if import_ortmodule_exception:
        raise import_ortmodule_exception

    return has_ortmodule, package_name, version, cuda_version


has_ortmodule, package_name, version, cuda_version = validate_build_package_info()
