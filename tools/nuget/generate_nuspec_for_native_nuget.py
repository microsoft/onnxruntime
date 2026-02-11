# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# These platform version values are the default target platform versions for .NET 9 from the table here:
# https://learn.microsoft.com/en-us/dotnet/standard/frameworks#os-version-in-tfms
platform_version_android = "35.0"
platform_version_ios = "18.0"
platform_version_maccatalyst = "18.0"


# What does the names of our C API tarball/zip files looks like
# os: win, linux, osx
# ep: cuda, tensorrt, None
def get_package_name(os, cpu_arch, ep, is_training_package):
    pkg_name = "onnxruntime-training" if is_training_package else "onnxruntime"
    if os == "win":
        pkg_name += "-win-"
        pkg_name += cpu_arch
    elif os == "linux":
        pkg_name += "-linux-"
        pkg_name += cpu_arch
    elif os == "osx":
        pkg_name = "onnxruntime-osx-" + cpu_arch
    return pkg_name


# Currently we take onnxruntime_providers_cuda from CUDA build
# And onnxruntime, onnxruntime_providers_shared and
# onnxruntime_providers_tensorrt from tensorrt build
# cuda binaries are split out into the platform dependent packages Microsoft.ML.OnnxRuntime.{Linux|Windows}
# and not included in the base Microsoft.ML.OnnxRuntime.Gpu package
def is_this_file_needed(ep, filename, package_name):
    if package_name == "Microsoft.ML.OnnxRuntime.Gpu":
        return False
    return (
        (ep != "cuda" or "cuda" in filename)
        and (ep != "tensorrt" or "cuda" not in filename)
        and (ep != "migraphx" or "migraphx" not in filename)
    )


# nuget_artifacts_dir: the directory with uncompressed C API tarball/zip files
# ep: cuda, tensorrt, None
# files_list: a list of xml string pieces to append
# This function has no return value. It updates files_list directly
def generate_file_list_for_ep(nuget_artifacts_dir, ep, files_list, include_pdbs, is_training_package, package_name):
    for child in nuget_artifacts_dir.iterdir():
        if not child.is_dir():
            continue

        for cpu_arch in ["x86", "x64", "arm", "arm64"]:
            if child.name == get_package_name("win", cpu_arch, ep, is_training_package):
                child = child / "lib"  # noqa: PLW2901
                for child_file in child.iterdir():
                    suffixes = [".dll", ".lib", ".pdb"] if include_pdbs else [".dll", ".lib"]
                    if (
                        child_file.suffix in suffixes
                        and is_this_file_needed(ep, child_file.name, package_name)
                        and package_name != "Microsoft.ML.OnnxRuntime.Gpu.Linux"
                    ):
                        files_list.append(
                            '<file src="' + str(child_file) + f'" target="runtimes/win-{cpu_arch}/native"/>'
                        )
        for cpu_arch in ["x86_64", "arm64"]:
            if child.name == get_package_name("osx", cpu_arch, ep, is_training_package):
                child = child / "lib"  # noqa: PLW2901
                if cpu_arch == "x86_64":
                    cpu_arch = "x64"  # noqa: PLW2901
                for child_file in child.iterdir():
                    # Check if the file has digits like onnxruntime.1.8.0.dylib. We can skip such things
                    is_versioned_dylib = re.match(r".*[\.\d+]+\.dylib$", child_file.name)
                    if child_file.is_file() and child_file.suffix == ".dylib" and not is_versioned_dylib:
                        files_list.append(
                            '<file src="' + str(child_file) + f'" target="runtimes/osx-{cpu_arch}/native"/>'
                        )
        for cpu_arch in ["x64", "aarch64"]:
            if child.name == get_package_name("linux", cpu_arch, ep, is_training_package):
                child = child / "lib"  # noqa: PLW2901
                if cpu_arch == "x86_64":
                    cpu_arch = "x64"  # noqa: PLW2901
                elif cpu_arch == "aarch64":
                    cpu_arch = "arm64"  # noqa: PLW2901
                for child_file in child.iterdir():
                    if not child_file.is_file():
                        continue
                    if (
                        child_file.suffix == ".so"
                        and is_this_file_needed(ep, child_file.name, package_name)
                        and package_name != "Microsoft.ML.OnnxRuntime.Gpu.Windows"
                    ):
                        files_list.append(
                            '<file src="' + str(child_file) + f'" target="runtimes/linux-{cpu_arch}/native"/>'
                        )

        if child.name == "onnxruntime-android" or child.name == "onnxruntime-training-android":
            for child_file in child.iterdir():
                if child_file.suffix in [".aar"]:
                    files_list.append('<file src="' + str(child_file) + '" target="runtimes/android/native"/>')

        if child.name == "onnxruntime-ios":
            for child_file in child.iterdir():
                if child_file.suffix in [".zip"]:
                    files_list.append('<file src="' + str(child_file) + '" target="runtimes/ios/native"/>')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime create nuget spec script (for hosting native shared library artifacts)",
        usage="",
    )
    # Main arguments
    parser.add_argument("--package_name", required=True, help="ORT package name. Eg: Qualcomm.ML.OnnxRuntime.QNN")
    parser.add_argument("--package_version", required=True, help="ORT package version. Eg: 1.0.0")
    parser.add_argument("--target_architecture", required=True, help="Eg: x64")
    parser.add_argument("--build_config", required=True, help="Eg: RelWithDebInfo")
    parser.add_argument("--ort_build_path", required=True, help="ORT build directory.")
    parser.add_argument("--native_build_path", required=True, help="Native build output directory.")
    parser.add_argument("--packages_path", required=True, help="Nuget packages output directory.")
    parser.add_argument("--sources_path", required=True, help="OnnxRuntime source code root.")
    parser.add_argument("--commit_id", required=True, help="The last commit id included in this package.")
    parser.add_argument(
        "--is_release_build",
        required=False,
        default=None,
        type=str,
        help="Flag indicating if the build is a release build. Accepted values: true/false.",
    )
    parser.add_argument(
        "--execution_provider",
        required=False,
        default="None",
        type=str,
        choices=["cuda", "dnnl", "openvino", "migraphx", "tensorrt", "snpe", "qnn", "None"],
        help="The selected execution provider for this build.",
    )
    parser.add_argument("--sdk_info", required=False, default="", type=str, help="dependency SDK information.")
    parser.add_argument(
        "--nuspec_name", required=False, default="NativeNuget.nuspec", type=str, help="nuget spec name."
    )

    return parser.parse_args()


def generate_id(line_list, package_name):
    line_list.append("<id>" + package_name + "</id>")


def generate_version(line_list, package_version):
    line_list.append("<version>" + package_version + "</version>")


def generate_authors(line_list, authors):
    line_list.append("<authors>" + authors + "</authors>")


def generate_owners(line_list, owners):
    line_list.append("<owners>" + owners + "</owners>")


def generate_description(line_list, package_name):
    description = ""

    if package_name == "Microsoft.AI.MachineLearning":
        description = "This package contains Windows ML binaries."
    elif "Microsoft.ML.OnnxRuntime.Training" in package_name:  # This is a Microsoft.ML.OnnxRuntime.Training.* package
        description = (
            "The onnxruntime-training native shared library artifacts are designed to efficiently train and infer "
            + "a wide range of ONNX models on edge devices, such as client machines, gaming consoles, and other "
            + "portable devices with a focus on minimizing resource usage and maximizing accuracy."
            + "See https://github.com/microsoft/onnxruntime-training-examples/tree/master/on_device_training for "
            + "more details."
        )
    elif "ML.OnnxRuntime" in package_name:  # This is a *.ML.OnnxRuntime.* package
        description = (
            "This package contains native shared library artifacts for all supported platforms of ONNX Runtime."
        )
    line_list.append("<description>" + description + "</description>")


def generate_copyright(line_list, copyright):
    line_list.append("<copyright>" + copyright + "</copyright>")


def generate_tags(line_list, tags):
    line_list.append("<tags>" + tags + "</tags>")


def generate_icon(line_list, icon_file):
    line_list.append("<icon>" + icon_file + "</icon>")


def generate_license(line_list):
    line_list.append('<license type="file">LICENSE</license>')


def generate_project_url(line_list, project_url):
    line_list.append("<projectUrl>" + project_url + "</projectUrl>")


def generate_repo_url(line_list, repo_url, commit_id):
    line_list.append('<repository type="git" url="' + repo_url + '"' + ' commit="' + commit_id + '" />')


def generate_readme(line_list):
    line_list.append("<readme>README.md</readme>")


def add_common_dependencies(xml_text, package_name, version):
    xml_text.append('<dependency id="Microsoft.ML.OnnxRuntime.Managed"' + ' version="' + version + '"/>')


def generate_dependencies(xml_text, package_name, version):
    dml_dependency = '<dependency id="Microsoft.AI.DirectML" version="1.15.4"/>'

    if package_name == "Microsoft.AI.MachineLearning":
        xml_text.append("<dependencies>")
        # Support .Net Core
        xml_text.append('<group targetFramework="net5.0">')
        xml_text.append(dml_dependency)
        xml_text.append("</group>")
        # UAP10.0.16299, This is the earliest release of the OS that supports .NET Standard apps
        xml_text.append('<group targetFramework="UAP10.0.16299">')
        xml_text.append(dml_dependency)
        xml_text.append("</group>")
        # Support Native C++
        xml_text.append('<group targetFramework="native">')
        xml_text.append(dml_dependency)
        xml_text.append("</group>")
        xml_text.append("</dependencies>")
    elif package_name != "Qualcomm.ML.OnnxRuntime.QNN":
        include_dml = package_name == "Microsoft.ML.OnnxRuntime.DirectML"

        xml_text.append("<dependencies>")

        # Support .Net Core
        xml_text.append('<group targetFramework="NETCOREAPP">')
        add_common_dependencies(xml_text, package_name, version)
        if include_dml:
            xml_text.append(dml_dependency)
        xml_text.append("</group>")
        # Support .Net Standard
        xml_text.append('<group targetFramework="NETSTANDARD">')
        add_common_dependencies(xml_text, package_name, version)
        if include_dml:
            xml_text.append(dml_dependency)
        xml_text.append("</group>")
        # Support .Net Framework
        xml_text.append('<group targetFramework="NETFRAMEWORK">')
        add_common_dependencies(xml_text, package_name, version)
        if include_dml:
            xml_text.append(dml_dependency)
        xml_text.append("</group>")
        # Support Native C++
        if include_dml:
            xml_text.append('<group targetFramework="native">')
            xml_text.append(dml_dependency)
            xml_text.append("</group>")

        xml_text.append("</dependencies>")


def get_env_var(key):
    return os.environ.get(key)


def generate_release_notes(line_list, dependency_sdk_info):
    line_list.append("<releaseNotes>")
    line_list.append("Release Def:")

    branch = get_env_var("BUILD_SOURCEBRANCH")
    line_list.append("\t" + "Branch: " + (branch if branch is not None else ""))

    version = get_env_var("BUILD_SOURCEVERSION")
    line_list.append("\t" + "Commit: " + (version if version is not None else ""))

    build_id = get_env_var("BUILD_BUILDID")
    line_list.append(
        "\t"
        + "Build: https://aiinfra.visualstudio.com/Lotus/_build/results?buildId="
        + (build_id if build_id is not None else "")
    )

    if dependency_sdk_info:
        line_list.append("Dependency SDK: " + dependency_sdk_info)

    line_list.append("</releaseNotes>")


def generate_metadata(line_list, args):
    tags = "native ONNX Runtime ONNXRuntime Machine Learning MachineLearning"
    if "Microsoft.ML.OnnxRuntime.Training." in args.package_name:
        tags.append(" ONNXRuntime-Training Learning-on-The-Edge On-Device-Training On-Device Training")

    metadata_list = ["<metadata>"]
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, "Microsoft")
    generate_owners(metadata_list, "Microsoft")
    generate_description(metadata_list, args.package_name)
    generate_copyright(metadata_list, "\xc2\xa9 " + "Microsoft Corporation. All rights reserved.")
    generate_tags(metadata_list, tags)
    generate_icon(metadata_list, "ORT_icon_for_light_bg.png")
    generate_license(metadata_list)
    generate_project_url(metadata_list, "https://github.com/onnxruntime/onnxruntime-qnn")
    generate_repo_url(metadata_list, "https://github.com/onnxruntime/onnxruntime-qnn.git", args.commit_id)
    generate_readme(metadata_list)
    generate_dependencies(metadata_list, args.package_name, args.package_version)
    generate_release_notes(metadata_list, args.sdk_info)
    metadata_list.append("</metadata>")

    line_list += metadata_list


def copy_file(src_path, dst_path):
    shutil.copyfile(src_path, dst_path)


def generate_files(line_list, args):
    files_list = ["<files>"]

    is_cpu_package = args.package_name in [
        "Microsoft.ML.OnnxRuntime",
        "Microsoft.ML.OnnxRuntime.Training",
    ]
    is_windowsai_package = args.package_name == "Microsoft.AI.MachineLearning"
    is_qnn_package = args.package_name == "Qualcomm.ML.OnnxRuntime.QNN"

    includes_winml = is_windowsai_package

    is_windows_build = is_windows()

    nuget_dependencies = {}

    if is_windows_build:
        nuget_dependencies = {
            "qnn_ep_shared_lib": "onnxruntime_providers_qnn.dll",
        }

        runtimes_target = '" target="runtimes\\win-'
    else:
        runtimes_target = '" target="runtimes\\linux-'

    if is_windowsai_package:
        runtimes_native_folder = "_native"
    else:
        runtimes_native_folder = "native"

    runtimes = f'{runtimes_target}{args.target_architecture}\\{runtimes_native_folder}"'

    # Process headers
    build_dir = "buildTransitive" if "Gpu" in args.package_name else "build"
    include_dir = f"{build_dir}\\native\\include"

    # Sub.Gpu packages do not include the onnxruntime headers
    if args.package_name != "Microsoft.ML.OnnxRuntime.Gpu" and args.package_name != "Microsoft.ML.OnnxRuntime.MIGraphX":
        files_list.append(
            "<file src="
            + '"'
            + os.path.join(args.sources_path, "include\\onnxruntime\\core\\session\\onnxruntime_*.h")
            + '" target="'
            + include_dir
            + '" />'
        )
        files_list.append(
            "<file src="
            + '"'
            + os.path.join(args.sources_path, "include\\onnxruntime\\core\\framework\\provider_options.h")
            + '" target="'
            + include_dir
            + '" />'
        )

    if is_qnn_package:
        files_list.append("<file src=" + '"' + os.path.join(args.native_build_path, "QnnCpu.dll") + runtimes + " />")
        files_list.append("<file src=" + '"' + os.path.join(args.native_build_path, "QnnHtp.dll") + runtimes + " />")
        if args.target_architecture != "x64":
            files_list.append(
                "<file src=" + '"' + os.path.join(args.native_build_path, "QnnGpu.dll") + runtimes + " />"
            )
            files_list.append(
                "<file src=" + '"' + os.path.join(args.native_build_path, "QnnSystem.dll") + runtimes + " />"
            )
            files_list.append(
                "<file src=" + '"' + os.path.join(args.native_build_path, "QnnHtpPrepare.dll") + runtimes + " />"
            )
            for htp_arch in [73, 81]:
                files_list.append(
                    "<file src="
                    + '"'
                    + os.path.join(args.native_build_path, f"QnnHtpV{htp_arch}Stub.dll")
                    + runtimes
                    + " />"
                )
                files_list.append(
                    "<file src="
                    + '"'
                    + os.path.join(args.native_build_path, f"libQnnHtpV{htp_arch}Skel.so")
                    + runtimes
                    + " />"
                )
                files_list.append(
                    "<file src="
                    + '"'
                    + os.path.join(args.native_build_path, f"libqnnhtpv{htp_arch}.cat")
                    + runtimes
                    + " />"
                )

    is_ado_packaging_build = False
    # Process runtimes
    # Process onnxruntime import lib, dll, and pdb
    # for Snpe android build
    if is_windows_build:
        nuget_artifacts_dir = Path(args.native_build_path) / "nuget-artifacts"
        # the winml package includes pdbs. for other packages exclude them.
        include_pdbs = includes_winml
        if nuget_artifacts_dir.exists():
            # Code path for ADO build pipeline, the files under 'nuget-artifacts' are
            # downloaded from other build jobs
            ep_list = [None]
            for ep in ep_list:
                generate_file_list_for_ep(nuget_artifacts_dir, ep, files_list, include_pdbs, False, args.package_name)
            is_ado_packaging_build = True

    # Process execution providers which are built as shared libs
    if args.execution_provider == "qnn" or (is_qnn_package and not is_ado_packaging_build):
        files_list.append(
            "<file src="
            + '"'
            + os.path.join(args.native_build_path, nuget_dependencies["qnn_ep_shared_lib"])
            + runtimes_target
            + args.target_architecture
            + '\\native" />'
        )

    # Process props and targets files
    if is_cpu_package or is_qnn_package:
        # Add QNN EP helper assembly for QNN packages
        if is_qnn_package:
            helper_dll_path = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Qualcomm.ML.OnnxRuntime.QNN",
                "bin",
                args.target_architecture,
                args.build_config,
                "netstandard2.0",
                "Qualcomm.ML.OnnxRuntime.QNN.dll",
            )

            if os.path.exists(helper_dll_path):
                files_list.append("<file src=" + '"' + helper_dll_path + '" target="lib\\netstandard2.0"  />')

        # Process props file
        if is_qnn_package:
            source_props = os.path.join(
                args.sources_path,
                "csharp",
                "src",
                "Microsoft.ML.OnnxRuntime",
                "targets",
                "netstandard",
                "props_qnn.xml",
            )
        else:
            source_props = os.path.join(
                args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "netstandard", "props.xml"
            )
        target_props = os.path.join(
            args.sources_path,
            "csharp",
            "src",
            "Microsoft.ML.OnnxRuntime",
            "targets",
            "netstandard",
            args.package_name + ".props",
        )
        copy_file(source_props, target_props)
        files_list.append("<file src=" + '"' + target_props + '" target="' + build_dir + '\\native" />')
        if not is_qnn_package:
            files_list.append("<file src=" + '"' + target_props + '" target="' + build_dir + '\\netstandard2.0"  />')
            files_list.append("<file src=" + '"' + target_props + '" target="' + build_dir + '\\netstandard2.1"  />')

        # Process targets file
        source_targets = os.path.join(
            args.sources_path, "csharp", "src", "Microsoft.ML.OnnxRuntime", "targets", "netstandard", "targets.xml"
        )
        target_targets = os.path.join(
            args.sources_path,
            "csharp",
            "src",
            "Microsoft.ML.OnnxRuntime",
            "targets",
            "netstandard",
            args.package_name + ".targets",
        )
        copy_file(source_targets, target_targets)
        files_list.append("<file src=" + '"' + target_targets + '" target="' + build_dir + '\\native"  />')
        if not is_qnn_package:
            files_list.append("<file src=" + '"' + target_targets + '" target="' + build_dir + '\\netstandard2.0" />')
            files_list.append("<file src=" + '"' + target_targets + '" target="' + build_dir + '\\netstandard2.1"  />')

    # README
    files_list.append(
        "<file src=" + '"' + os.path.join(args.sources_path, "tools/nuget/nupkg.README.md") + '" target="README.md" />'
    )

    # Process License, ThirdPartyNotices, Privacy
    files_list.append("<file src=" + '"' + os.path.join(args.sources_path, "LICENSE") + '" target="LICENSE" />')
    files_list.append(
        "<file src="
        + '"'
        + os.path.join(args.sources_path, "ThirdPartyNotices.txt")
        + '" target="ThirdPartyNotices.txt" />'
    )
    files_list.append(
        "<file src=" + '"' + os.path.join(args.sources_path, "docs", "Privacy.md") + '" target="Privacy.md" />'
    )
    files_list.append(
        "<file src="
        + '"'
        + os.path.join(args.sources_path, "ORT_icon_for_light_bg.png")
        + '" target="ORT_icon_for_light_bg.png" />'
    )
    if is_qnn_package:
        files_list.append(
            "<file src="
            + '"'
            + os.path.join(args.native_build_path, "Qualcomm_LICENSE.pdf")
            + '" target="Qualcomm_LICENSE.pdf" />'
        )
    files_list.append("</files>")

    line_list += files_list


def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append("<package>")
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append("</package>")
    return lines


def is_windows():
    return sys.platform.startswith("win")


def is_linux():
    return sys.platform.startswith("linux")


def is_macos():
    return sys.platform.startswith("darwin")


def validate_platform():
    if not (is_windows() or is_linux() or is_macos()):
        raise Exception("Native Nuget generation is currently supported only on Windows, Linux, and MacOS")


def validate_execution_provider(execution_provider):
    if is_linux():
        if not (
            execution_provider == "None"
            or execution_provider == "dnnl"
            or execution_provider == "cuda"
            or execution_provider == "tensorrt"
            or execution_provider == "openvino"
            or execution_provider == "migraphx"
        ):
            raise Exception(
                "On Linux platform nuget generation is supported only "
                "for cpu|cuda|dnnl|tensorrt|openvino|migraphx execution providers."
            )


def main():
    # Parse arguments
    args = parse_arguments()

    validate_platform()

    validate_execution_provider(args.execution_provider)

    if args.is_release_build.lower() != "true" and args.is_release_build.lower() != "false":
        raise Exception("Only valid options for IsReleaseBuild are: true and false")

    # Generate nuspec
    lines = generate_nuspec(args)

    # Create the nuspec needed to generate the Nuget
    print(f"nuspec_name: {args.nuspec_name}")
    with open(os.path.join(args.native_build_path, args.nuspec_name), "w") as f:
        for line in lines:
            # Uncomment the printing of the line if you need to debug what's produced on a CI machine
            # print(line)
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
