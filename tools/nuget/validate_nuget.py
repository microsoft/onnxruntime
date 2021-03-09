# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import os
import zipfile  # Available Python 3.2 or higher
import glob


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Validate ONNX Runtime native nuget containing native shared library artifacts spec script",
        usage='')
    # Main arguments
    parser.add_argument("--nuget_package", required=True, help="Nuget package name to be validated.")
    parser.add_argument("--nuget_path", required=True,
                        help="Path containing the Nuget to be validated. Must only contain only one Nuget within this.")
    parser.add_argument("--platforms_supported", required=True,
                        help="Comma separated list (no space). Ex: linux-x64,win-x86,osx-x64")
    parser.add_argument("--verify_nuget_signing", required=True,
                        help="Flag indicating if Nuget package signing is to be verified. "
                             "Only accepts 'true' or 'false'")

    return parser.parse_args()


def check_exists(path):
    return os.path.exists(path)


def is_windows():
    return sys.platform.startswith("win")


def check_if_dlls_are_present(is_windows_ai_package, platforms_supported, zip_file):
    platforms = platforms_supported.strip().split(",")
    for platform in platforms:
        if platform.startswith("win"):
            native_folder = '_native' if is_windows_ai_package else 'native'
            path = "runtimes/" + platform + "/" + native_folder + "/onnxruntime.dll"
            print('Checking path: ' + path)
            if (path not in zip_file.namelist()):
                print("onnxruntime.dll not found for " + platform)
                print(zip_file.namelist())
                raise Exception("onnxruntime.dll not found for " + platform)

        elif platform.startswith("linux"):
            path = "runtimes/" + platform + "/native/libonnxruntime.so"
            print('Checking path: ' + path)
            if (path not in zip_file.namelist()):
                print("libonnxruntime.so not found for " + platform)
                raise Exception("libonnxruntime.so not found for " + platform)

        elif platform.startswith("osx"):
            path = "runtimes/" + platform + "/native/libonnxruntime.dylib"
            print('Checking path: ' + path)
            if (path not in zip_file.namelist()):
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


def main():
    args = parse_arguments()

    files = glob.glob(os.path.join(args.nuget_path, args.nuget_package))
    nuget_packages_found_in_path = [i for i in files if i.endswith('.nupkg') and "Managed" not in i]
    if (len(nuget_packages_found_in_path) != 1):
        print('Nuget packages found in path: ')
        print(nuget_packages_found_in_path)
        raise Exception('No Nuget packages / more than one Nuget packages found in the given path.')
    nuget_file_name = nuget_packages_found_in_path[0]
    full_nuget_path = os.path.join(args.nuget_path, nuget_file_name)

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
        check_if_dlls_are_present(is_windows_ai_package, args.platforms_supported, zip_file)

        # Check if the Nuget has been signed
        if (args.verify_nuget_signing != 'true' and args.verify_nuget_signing != 'false'):
            raise Exception('Parameter verify_nuget_signing accepts only true or false as an argument')

        if (args.verify_nuget_signing == 'true'):
            print('Verifying if Nuget has been signed')
            if(not check_if_nuget_is_signed(args.nuget_path)):
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


if __name__ == "__main__":
    sys.exit(main())
