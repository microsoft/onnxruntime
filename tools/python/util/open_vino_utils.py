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


def openvino_verify_device_type(device_read):
    choices = ["CPU_FP32", "CPU_FP16", "GPU_FP32", "GPU_FP16"]

    choices1 = [
        "CPU_FP32_NO_PARTITION",
        "CPU_FP16_NO_PARTITION",
        "GPU_FP32_NO_PARTITION",
        "GPU_FP16_NO_PARTITION",
    ]
    status_hetero = True
    res = False
    if device_read in choices:
        res = True
    elif device_read in choices1:
        res = True
    elif device_read.startswith("HETERO:") or device_read.startswith("MULTI:") or device_read.startswith("AUTO:"):
        res = True
        comma_separated_devices = device_read.split(":")
        comma_separated_devices = comma_separated_devices[1].split(",")
        if len(comma_separated_devices) < 2:
            print("At least two devices required in Hetero/Multi/Auto Mode")
            status_hetero = False
        dev_options = ["CPU", "GPU"]
        for dev in comma_separated_devices:
            if dev not in dev_options:
                status_hetero = False
                break

    def invalid_hetero_build():
        print("\nIf trying to build Hetero/Multi/Auto, specify the supported devices along with it.\n")
        print("specify the keyword HETERO or MULTI or AUTO followed by the devices ")
        print("in the order of priority you want to build\n")
        print("The different hardware devices that can be added in HETERO or MULTI or AUTO")
        print("are ['CPU','GPU'] \n")
        print("An example of how to specify the hetero build type. Ex: HETERO:GPU,CPU \n")
        print("An example of how to specify the MULTI build type. Ex: MULTI:GPU,CPU \n")
        print("An example of how to specify the AUTO build type. Ex: AUTO:GPU,CPU \n")
        sys.exit("Wrong Build Type selected")

    if res is False:
        print("\nYou have selected wrong configuration for the build.")
        print("pick the build type for specific Hardware Device from following options: ", choices)
        print("(or) from the following options with graph partitioning disabled: ", choices1)
        print("\n")
        if not (device_read.startswith("HETERO") or device_read.startswith("MULTI") or device_read.startswith("AUTO")):
            invalid_hetero_build()
        sys.exit("Wrong Build Type selected")

    if status_hetero is False:
        invalid_hetero_build()

    return device_read
