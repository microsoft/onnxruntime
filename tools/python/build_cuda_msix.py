

import os
import shutil
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

import subprocess

"""
Builds the CUDA Dependencies MSIX Package for ONNX Runtime.
The package includes the necessary CUDA DLLs for the CUDA Execution Provider.
  - "cublas64_*.dll"
  - "cublasLt64_*.dll"
  - "cudart64_*.dll"
  - "cudnn64_*.dll"
  - "cudnn_graph64_*.dll"
  - "cudnn_ops64_*.dll"
  - "cufft64_*.dll"
  - "onnxruntime_providers_cuda.dll"
  - "onnxruntime-genai-cuda.dll"

Example Usage:

$ python build_cuda_package.py --cuda_deps_directory c:\\cuda_deps --output_directory c:\\cudamsix --version 1.0.0.1 --platform_version 1.0.0 --clean

Testing unsigned packages requires running as administrator. See
https://learn.microsoft.com/en-us/windows/msix/package/unsigned-package for more details.

# From powershell, from the base of this repo:
python build_cuda_package.py --cuda_deps_directory c:\\cuda_deps --version 1.0.0.1 --platform_version 1.0.0 --clean --build_unsigned_package
Get-AppxPackage -Name Microsoft.CUDAExecutionProvider | Remove-AppxPackage
sudo powershell -Command "Add-AppxPackage -Path $pwd\\artifacts\\cuda-msix\\CUDAEP-x64-1.0.0.msix -AllowUnsigned -ForceUpdateFromAnyVersion; Read-Host 'Press Enter to continue...'"
"""

def run(
    *args,
    cwd=None,
    input=None,
    capture_stdout=False,
    capture_stderr=False,
    shell=False,
    env=None,
    check=True,
    quiet=False,
):
    """Runs a subprocess.

    Args:
        *args: The subprocess arguments.
        cwd: The working directory. If None, specifies the current directory.
        input: The optional input byte sequence.
        capture_stdout: Whether to capture stdout.
        capture_stderr: Whether to capture stderr.
        shell: Whether to run using the shell.
        env: The environment variables as a dict. If None, inherits the current
            environment.
        check: Whether to raise an error if the return code is not zero.
        quiet: If true, do not print output from the subprocess.

    Returns:
        A subprocess.CompletedProcess instance.
    """
    cmd = [*args]

    print(
        "Running subprocess in '{}'\n  {}".format(cwd or os.getcwd(), " ".join([shlex.quote(arg) for arg in cmd]))
    )

    def output(is_stream_captured):
        return subprocess.PIPE if is_stream_captured else (subprocess.DEVNULL if quiet else None)

    completed_process = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        input=input,
        stdout=output(capture_stdout),
        stderr=output(capture_stderr),
        env=env,
        shell=shell,
    )
    return completed_process

def run_subprocess(
    args,
    cwd=None,
    capture_stdout=False,
    shell=False,
    env=None,
):
    if env is None:
        env = {}
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    print(" ".join(args))
    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell)

def create_msix_package(app_manifest: Path, mapping_file: Path, output_package: Path):
    # Command to create MSIX package using makeappx.exe
    command = ["MakeAppx", "pack", "/m", app_manifest.as_posix(), "/f", mapping_file.as_posix(), "/p", output_package.as_posix()]

    # Print the command to be executed
    print(f"Executing command: {' '.join(command)}")

    # Execute the command
    result = run(command)

    # Print the stderr output
    if result.stderr:
        print(f"stderr: {result.stderr}")

    if result.returncode == 0:
        print(f"The MSIX package has been created successfully at {output_package}.")
    else:
        print(f"Failed to create MSIX package. Error: {result.stderr}")

def copy_cuda_dependencies(source_directory: Path, target_directory: Path):
    """Copy CUDA DLLs to the target directory."""

    # CUDA DLLs that should be included in the root
    cuda_dll_patterns = [
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "cudart64_*.dll",
        "cudnn64_*.dll",
        "cudnn_graph64_*.dll",
        "cudnn_ops64_*.dll",
        "cufft64_*.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime-genai-cuda.dll",
    ]

    target_directory.mkdir(parents=True, exist_ok=True)

    # Create ExecutionProvider subdirectory and copy all CUDA DLLs there
    execution_provider_dir = target_directory / "ExecutionProvider"
    execution_provider_dir.mkdir(parents=True, exist_ok=True)

    # Copy all CUDA DLLs to ExecutionProvider folder as well
    print(f"\nCopying CUDA DLLs to ExecutionProvider folder...")
    for pattern in cuda_dll_patterns:
        for file_path in source_directory.glob(pattern):
            if file_path.is_file():
                dest_path = execution_provider_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                print(f"Copied {file_path.name} to {execution_provider_dir}")


def generate_appxmanifest(target_folder: Path, version: str, platform_version: str, build_unsigned_package: bool, cpu_arch: str):
    """Generate the AppxManifest.xml file for the CUDA EP MSIX package."""

    # Create the root element with all required namespaces
    package = ET.Element("Package", {
        "xmlns": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
        "xmlns:uap": "http://schemas.microsoft.com/appx/manifest/uap/windows10",
        "xmlns:uap10": "http://schemas.microsoft.com/appx/manifest/uap/windows10/10",
        "xmlns:uap15": "http://schemas.microsoft.com/appx/manifest/uap/windows10/15",
        "xmlns:uap17": "http://schemas.microsoft.com/appx/manifest/uap/windows10/17",
        "IgnorableNamespaces": "uap uap10 uap15 uap17"
    })

    publisher_string = "CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, S=Washington, C=US"
    if build_unsigned_package:
        # The string is from https://learn.microsoft.com/en-us/windows/msix/package/unsigned-package
        publisher_string += ", OID.2.25.311729368913984317654407730594956997722=1"

    additional_info = ""
    platform_version_prerelease_tag = platform_version.find('-')
    if platform_version_prerelease_tag != -1:
        # If the version contains a build metadata suffix like -prerelease, we strip it for the Identity element
        additional_info = f" ({platform_version[platform_version_prerelease_tag + 1:]})"

    # Create the Identity element
    identity = ET.SubElement(package, "Identity", {
        "Name": "Microsoft.FoundryLocal.CUDA.EP",
        "Version": version,
        "Publisher": publisher_string,
        "ProcessorArchitecture": cpu_arch
    })

    # Create the Properties element
    properties = ET.SubElement(package, "Properties")
    ET.SubElement(properties, "DisplayName").text = f"CUDA Execution Provider{additional_info}"
    ET.SubElement(properties, "PublisherDisplayName").text = "Microsoft Corporation"
    ET.SubElement(properties, "Description").text = f"CUDA Execution Provider{additional_info}"
    ET.SubElement(properties, "Logo").text = "images\\icon.png"

    # Add DependencyTarget element
    ET.SubElement(properties, "{http://schemas.microsoft.com/appx/manifest/uap/windows10/15}DependencyTarget").text = "true"

    # Add PackageIntegrity element
    package_integrity = ET.SubElement(properties, "{http://schemas.microsoft.com/appx/manifest/uap/windows10/10}PackageIntegrity")
    ET.SubElement(package_integrity, "{http://schemas.microsoft.com/appx/manifest/uap/windows10/10}Content", {"Enforcement": "on"})

    # Create the Resources element
    resources = ET.SubElement(package, "Resources")
    ET.SubElement(resources, "Resource", {"Language": "en-us"})

    # Create the Dependencies element
    dependencies = ET.SubElement(package, "Dependencies")
    ET.SubElement(dependencies, "TargetDeviceFamily", {
        "Name": "Windows.Desktop",
        "MinVersion": "10.0.19041.0",
        "MaxVersionTested": "10.0.26100.0"
    })

    # Create the Extensions element
    extensions = ET.SubElement(package, "Extensions")

    # Add uap17:Extension for package extension
    uap17_extension = ET.SubElement(extensions, "{http://schemas.microsoft.com/appx/manifest/uap/windows10/17}Extension", {
        "Category": "windows.packageExtension"
    })

    # Add PackageExtension element
    package_extension = ET.SubElement(uap17_extension, "{http://schemas.microsoft.com/appx/manifest/uap/windows10/17}PackageExtension", {
        "Name": "com.microsoft.cuda.executionprovider",
        "Id": "CUDAEP",
        "DisplayName": "CUDA Execution Provider",
        "PublicFolder": "ExecutionProvider",
        "Description": "CUDA Execution Provider"
    })

    # Add Properties inside PackageExtension
    extension_properties = ET.SubElement(package_extension, "{http://schemas.microsoft.com/appx/manifest/uap/windows10/17}Properties")
    provider_name = ET.SubElement(extension_properties, "ProviderName")
    provider_name.text = "CUDAExecutionProvider"
    provider_path = ET.SubElement(extension_properties, "ProviderPath")
    provider_path.text = "onnxruntime_providers_cuda.dll"

    # Create the target directory if it does not exist
    target_folder.mkdir(parents=True, exist_ok=True)

    # Write the content to the appxmanifest.xml file
    tree = ET.ElementTree(package)
    ET.indent(tree, space="  ", level=0)
    tree.write(target_folder / "appxmanifest.xml", encoding="utf-8", xml_declaration=True)
    print(f"Generated appxmanifest.xml at {target_folder / 'appxmanifest.xml'}")

# Note: generate_mapping_txt and create_msix_package are imported from build_package.py

def main():
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build CUDA EP MSIX package script")
    parser.add_argument("--cuda_deps_directory", type=Path, required=True, help="The directory containing CUDA DLLs")
    parser.add_argument("--output_directory", type=Path, default=get_repo_root() / "artifacts" / "cuda-msix", help="The output directory for the MSIX package")
    parser.add_argument("--version", type=str, required=True, help="The version number for the appxmanifest (MUST be in the format x.y.z.w where x, y, z, w are integers)")
    parser.add_argument("--platform_version", type=str, required=False, help="The version number for the installer name (valid version strings possibly with a prerelease tag)")
    parser.add_argument("--build_unsigned_package", action="store_true", help="Build a dev package that can be installed without having a codesign certificate")
    parser.add_argument("--build_dev_package", action="store_true", help="Build a dev package (staging only, no MSIX creation)")
    parser.add_argument("--clean", action="store_true", help="Clean the output directory before building")

    args = parser.parse_args()

    if not args.cuda_deps_directory.exists():
        raise RuntimeError(f"The cuda_deps_directory does not exist: {args.cuda_deps_directory}")

    images_folder = Path(SCRIPT_DIR) / "images"

    # Populate the staging path
    if args.clean:
        shutil.rmtree(args.output_directory, ignore_errors=True)

    package_staging_path = args.output_directory / "staging"
    package_staging_path.mkdir(parents=True, exist_ok=True)

    add_unsigned_oid = False
    if args.build_unsigned_package and args.build_dev_package:
        raise RuntimeError("You cannot build both an unsigned and a dev package at the same time.")
    elif args.build_unsigned_package:
        add_unsigned_oid = True
        print("Building unsigned package.")

    platform_version = args.platform_version
    if not platform_version:
        platform_version = args.version

    # Only build for x64 architecture (CUDA is not available on ARM64)
    cpu_arch = "x64"

    subfolder = f"cuda_win-{cpu_arch}"
    bin_folder = package_staging_path / subfolder
    bin_folder.mkdir(parents=True, exist_ok=True)

    # Copy CUDA dependencies
    print(f"Copying CUDA dependencies from {args.cuda_deps_directory} to {bin_folder}")
    copy_cuda_dependencies(args.cuda_deps_directory, bin_folder)

    # Copy images folder if it exists
    if images_folder.exists():
        shutil.copytree(images_folder, bin_folder / "images", dirs_exist_ok=True)
        print(f"Copied images from {images_folder} to {bin_folder / 'images'}")

    # Generate the appxmanifest.xml file
    generate_appxmanifest(bin_folder, args.version, platform_version, add_unsigned_oid, cpu_arch)

    # Generate the Mapping.txt file
    generate_mapping_txt(bin_folder)

    if not args.build_dev_package:
        msix_output_path = args.output_directory / f"CUDAEP-{platform_version}.msix"
        create_msix_package(
            app_manifest=bin_folder / "appxmanifest.xml",
            mapping_file=bin_folder / "Mapping.txt",
            output_package=msix_output_path)

        print(f"The MSIX package has been created at {msix_output_path}.")
    else:
        print(f"Dev package staging completed. Register with: Add-AppxPackage -Path {bin_folder / 'appxmanifest.xml'} -Register")

if __name__ == "__main__":
    main()
