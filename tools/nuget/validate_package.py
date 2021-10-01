# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import os
import zipfile  # Available Python 3.2 or higher
import glob
import re

linux_gpu_package_libraries = ["libonnxruntime_providers_shared.so", "libonnxruntime_providers_cuda.so",
                               "libonnxruntime_providers_tensorrt.so"]
win_gpu_package_libraries = ["onnxruntime_providers_shared.lib", "onnxruntime_providers_shared.dll",
                             "onnxruntime_providers_cuda.lib", "onnxruntime_providers_cuda.dll",
                             "onnxruntime_providers_tensorrt.lib", "onnxruntime_providers_tensorrt.dll"]
gpu_related_header_files = ["cpu_provider_factory.h", "tensorrt_provider_factory.h", "onnxruntime_c_api.h",
                            "onnxruntime_cxx_api.h", "onnxruntime_cxx_inline.h"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validate ONNX Runtime native nuget containing native shared library artifacts spec script",
        usage='')
    # Main arguments
    parser.add_argument("--package_type", required=True, help="Specify nuget, tarball or zip.")
    parser.add_argument("--package_name", required=True, help="Package name to be validated.")
    parser.add_argument("--package_path", required=True, help="Path containing the package to be validated." +
                        "Must only contain only one package within this.")
    parser.add_argument("--platforms_supported", required=True,
                        help="Comma separated list (no space). Ex: linux-x64,win-x86,osx-x64")
    parser.add_argument("--verify_nuget_signing",
                        help="Flag indicating if Nuget package signing is to be verified. "
                             "Only accepts 'true' or 'false'")

    return parser.parse_args()


def check_exists(path):
    return os.path.exists(path)


def is_windows():
    return sys.platform.startswith("win")


def check_if_dlls_are_present(package_type, is_windows_ai_package, is_gpu_package, platforms_supported,
                              zip_file, package_path):
    platforms = platforms_supported.strip().split(",")
    if package_type == "tarball":
        file_list_in_package = list()
        for (dirpath, dirnames, filenames) in os.walk(package_path):
            file_list_in_package += [os.path.join(dirpath, file) for file in filenames]
    else:
        file_list_in_package = zip_file.namelist()

    for platform in platforms:
        if platform.startswith("win"):
            native_folder = '_native' if is_windows_ai_package else 'native'

            if package_type == "nuget":
                folder = "runtimes/" + platform + "/" + native_folder
                header_folder = "build/native/include"
            else:  # zip package
                folder = package_path + "/lib"
                header_folder = package_path + "/include"

            path = folder + "/" + "onnxruntime.dll"
            print('Checking path: ' + path)
            if (path not in file_list_in_package):
                print("onnxruntime.dll not found for " + platform)
                raise Exception("onnxruntime.dll not found for " + platform)

            if is_gpu_package:
                for dll in win_gpu_package_libraries:
                    path = folder + "/" + dll
                    print('Checking path: ' + path)
                    if (path not in file_list_in_package):
                        print(dll + " not found for " + platform)
                        raise Exception(dll + " not found for " + platform)
                for header in gpu_related_header_files:
                    path = header_folder + "/" + header
                    print('Checking path: ' + path)
                    if (path not in file_list_in_package):
                        print(header + " not found for " + platform)
                        raise Exception(header + " not found for " + platform)

        elif platform.startswith("linux"):
            if package_type == "nuget":
                folder = "runtimes/" + platform + "/native"
                header_folder = "build/native/include"
            else:  # tarball package
                folder = package_path + "/lib"
                header_folder = package_path + "/include"

            path = folder + "/" + "libonnxruntime.so"
            print('Checking path: ' + path)
            if (path not in file_list_in_package):
                print("libonnxruntime.so not found for " + platform)
                raise Exception("libonnxruntime.so not found for " + platform)

            if is_gpu_package:
                for so in linux_gpu_package_libraries:
                    path = folder + "/" + so
                    print('Checking path: ' + path)
                    if (path not in file_list_in_package):
                        print(so + " not found for " + platform)
                        raise Exception(so + " not found for " + platform)
                for header in gpu_related_header_files:
                    path = header_folder + "/" + header
                    print('Checking path: ' + path)
                    if (path not in file_list_in_package):
                        print(header + " not found for " + platform)
                        raise Exception(header + " not found for " + platform)

        elif platform.startswith("osx"):
            path = "runtimes/" + platform + "/native/libonnxruntime.dylib"
            print('Checking path: ' + path)
            if (path not in file_list_in_package):
                print("libonnxruntime.dylib not found for " + platform)
                raise Exception("libonnxruntime.dylib not found for " + platform)

        else:
            raise Exception("Unsupported platform: " + platform)


def check_if_nuget_is_signed(nuget_path):
    code_sign_summary_file = glob.glob(os.path.join(nuget_path, '*.md'))
    if (len(code_sign_summary_file) != 1):
        print('CodeSignSummary files found in path: ')
        print(code_sign_summary_file)
        raise Exception('No CodeSignSummary files / more than one CodeSignSummary files found in the given path.')

    print('CodeSignSummary file: ' + code_sign_summary_file[0])

    with open(code_sign_summary_file[0]) as f:
        contents = f.read()
        return "Pass" in contents

    return False


def validate_tarball(args):
    files = glob.glob(os.path.join(args.package_path, args.package_name))
    if (len(files) != 1):
        print('packages found in path: ')
        print(files)
        raise Exception('No packages / more than one packages found in the given path.')

    package_name = args.package_name
    if "-gpu-" in package_name.lower():
        is_gpu_package = True
    else:
        is_gpu_package = False

    package_folder = re.search('(.*)[.].*', package_name).group(1)

    print('tar zxvf ' + package_name)
    os.system("tar zxvf " + package_name)

    is_windows_ai_package = False
    zip_file = None
    check_if_dlls_are_present(args.package_type, is_windows_ai_package, is_gpu_package,
                              args.platforms_supported, zip_file, package_folder)


def validate_zip(args):
    files = glob.glob(os.path.join(args.package_path, args.package_name))
    if (len(files) != 1):
        print('packages found in path: ')
        print(files)
        raise Exception('No packages / more than one packages found in the given path.')

    package_name = args.package_name
    if "-gpu-" in package_name.lower():
        is_gpu_package = True
    else:
        is_gpu_package = False

    package_folder = re.search('(.*)[.].*', package_name).group(1)

    is_windows_ai_package = False
    zip_file = zipfile.ZipFile(package_name)
    check_if_dlls_are_present(args.package_type, is_windows_ai_package, is_gpu_package,
                              args.platforms_supported, zip_file, package_folder)


def validate_nuget(args):
    files = glob.glob(os.path.join(args.package_path, args.package_name))
    nuget_packages_found_in_path = [i for i in files if i.endswith('.nupkg') and "Managed" not in i]
    if (len(nuget_packages_found_in_path) != 1):
        print('Nuget packages found in path: ')
        print(nuget_packages_found_in_path)
        raise Exception('No Nuget packages / more than one Nuget packages found in the given path.')
    nuget_file_name = nuget_packages_found_in_path[0]
    full_nuget_path = os.path.join(args.package_path, nuget_file_name)

    if "Gpu" in nuget_file_name:
        is_gpu_package = True
    else:
        is_gpu_package = False

    exit_code = 0

    nupkg_copy_name = "NugetCopy" + ".nupkg"
    zip_copy_name = "NugetCopy" + ".zip"
    zip_file = None

    # Remove any residual files
    if check_exists(nupkg_copy_name):
        os.remove(nupkg_copy_name)

    if check_exists(zip_copy_name):
        os.remove(zip_copy_name)

    # Do all validations here
    try:
        if not is_windows():
            raise Exception('Nuget validation is currently supported only on Windows')

        # Make a copy of the Nuget package
        print('Copying [' + full_nuget_path + '] -> [' + nupkg_copy_name + '], and extracting its contents')
        os.system("copy " + full_nuget_path + " " + nupkg_copy_name)

        # Convert nupkg to zip
        os.rename(nupkg_copy_name, zip_copy_name)
        zip_file = zipfile.ZipFile(zip_copy_name)

        # Check if the relevant dlls are present in the Nuget/Zip
        print('Checking if the Nuget contains relevant dlls')
        is_windows_ai_package = os.path.basename(full_nuget_path).startswith('Microsoft.AI.MachineLearning')
        check_if_dlls_are_present(args.package_type, is_windows_ai_package, is_gpu_package,
                                  args.platforms_supported, zip_file, None)

        # Check if the Nuget has been signed
        if (args.verify_nuget_signing != 'true' and args.verify_nuget_signing != 'false'):
            raise Exception('Parameter verify_nuget_signing accepts only true or false as an argument')

        if (args.verify_nuget_signing == 'true'):
            print('Verifying if Nuget has been signed')
            if(not check_if_nuget_is_signed(args.package_path)):
                print('Nuget signing verification failed')
                raise Exception('Nuget signing verification failed')

    except Exception as e:
        print(e)
        exit_code = 1

    finally:
        print('Cleaning up after Nuget validation')

        if zip_file is not None:
            zip_file.close()

        if check_exists(zip_copy_name):
            os.remove(zip_copy_name)

        if exit_code == 0:
            print('Nuget validation was successful')
        else:
            raise Exception('Nuget validation was unsuccessful')


def main():
    args = parse_arguments()

    if args.package_type == "nuget":
        validate_nuget(args)
    elif args.package_type == "tarball":
        validate_tarball(args)
    elif args.package_type == "zip":
        validate_zip(args)
    else:
        print('Package type {} is not supported'.format(args.package_type))


if __name__ == "__main__":
    sys.exit(main())
