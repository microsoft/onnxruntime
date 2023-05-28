# Copyright (C) 2022-2023 Intel Corporation
# Licensed under the MIT License

import os
import site
import sys


def add_openvino_libs_to_path() -> None:
    """Adds OpenVINO libraries to the PATH environment variable on Windows."""
    if sys.platform == "win32":
        # Installer, pip installs openvino dlls to the different directories
        # and those paths need to be visible to the openvino-ep modules
        #
        # If you're using a custom installation of openvino,
        # add the location of openvino dlls to your system PATH.
        openvino_libs = []
        # looking for the libs in the pip installation path.
        if os.path.isdir(os.path.join(site.getsitepackages()[1], "openvino", "libs")):
            openvino_libs.append(os.path.join(site.getsitepackages()[1], "openvino", "libs"))
        else:
            # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
            openvino_libs_installer = os.getenv("OPENVINO_LIB_PATHS")
            if openvino_libs_installer:
                openvino_libs.extend(openvino_libs_installer.split(";"))
            else:
                sys.exit(
                    "Error: Please set the OPENVINO_LIB_PATHS environment variable. "
                    "If you use an install package, please, run setupvars.bat"
                )
        for lib in openvino_libs:
            lib_path = os.path.join(os.path.dirname(__file__), lib)
            if os.path.isdir(lib_path):
                os.environ["PATH"] = os.path.abspath(lib_path) + ";" + os.environ["PATH"]
                os.add_dll_directory(os.path.abspath(lib_path))
