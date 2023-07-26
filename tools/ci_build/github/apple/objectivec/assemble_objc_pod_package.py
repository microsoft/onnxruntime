#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import pathlib
import sys

_script_dir = pathlib.Path(__file__).parent.resolve(strict=True)
sys.path.append(str(_script_dir.parent))


from c.assemble_c_pod_package import get_pod_config_file as get_c_pod_config_file  # noqa: E402
from package_assembly_utils import (  # noqa: E402
    PackageVariant,
    copy_repo_relative_to_dir,
    filter_files,
    gen_file_from_template,
    load_json_config,
)

# these variables contain paths or path patterns that are relative to the repo root

# the license file
license_file = "LICENSE"

# include directories for compiling the pod itself
include_dirs = [
    "objectivec",
]

all_objc_files = {
    "source_files": [
        "objectivec/include/*.h",
        "objectivec/*.h",
        "objectivec/*.m",
        "objectivec/*.mm",
    ],
    "public_header_files": [
        "objectivec/include/*.h",
    ],
    "test_source_files": [
        "objectivec/test/*.h",
        "objectivec/test/*.m",
        "objectivec/test/*.mm",
    ],
    "test_resource_files": [
        "objectivec/test/testdata/*.ort",
        "onnxruntime/test/testdata/training_api/*.onnx",
        "onnxruntime/test/testdata/training_api/*.ckpt",
        "onnxruntime/test/testdata/training_api/*.in",
        "onnxruntime/test/testdata/training_api/*.out",
    ],
}

training_only_objc_files = {
    "source_files": [
        "objectivec/include/onnxruntime_training.h",
        "objectivec/include/ort_checkpoint.h",
        "objectivec/include/ort_training_session.h",
        "objectivec/ort_checkpoint.mm",
        "objectivec/ort_checkpoint_internal.h",
        "objectivec/ort_training_session_internal.h",
        "objectivec/ort_training_session.mm",
    ],
    "public_header_files": [
        "objectivec/include/ort_checkpoint.h",
        "objectivec/include/ort_training_session.h",
        "objectivec/include/onnxruntime_training.h",
    ],
    "test_source_files": [
        "objectivec/test/ort_training_session_test.mm",
        "objectivec/test/ort_checkpoint_test.mm",
        "objectivec/test/ort_training_utils_test.mm",
    ],
    "test_resource_files": [
        "onnxruntime/test/testdata/training_api/*.onnx",
        "onnxruntime/test/testdata/training_api/*.ckpt",
        "onnxruntime/test/testdata/training_api/*.in",
        "onnxruntime/test/testdata/training_api/*.out",
    ],
}


def get_pod_files(package_variant: PackageVariant):
    """
    Gets the source and header files for the given package variant.
    """
    if package_variant == PackageVariant.Training:
        return all_objc_files
    else:
        # return files that are in pod_files but not in training_only_objc_files
        filtered_pod_files = {}
        for key in all_objc_files:
            filtered_pod_files[key] = filter_files(all_objc_files[key], training_only_objc_files[key])
        return filtered_pod_files


def get_pod_config_file(package_variant: PackageVariant):
    """
    Gets the pod configuration file path for the given package variant.
    """
    if package_variant == PackageVariant.Full:
        return _script_dir / "onnxruntime-objc.config.json"
    elif package_variant == PackageVariant.Mobile:
        return _script_dir / "onnxruntime-mobile-objc.config.json"
    elif package_variant == PackageVariant.Training:
        return _script_dir / "onnxruntime-training-objc.config.json"
    else:
        raise ValueError(f"Unhandled package variant: {package_variant}")


def assemble_objc_pod_package(
    staging_dir: pathlib.Path, pod_version: str, framework_info_file: pathlib.Path, package_variant: PackageVariant
):
    """
    Assembles the files for the Objective-C pod package in a staging directory.

    :param staging_dir Path to the staging directory for the Objective-C pod files.
    :param pod_version Objective-C pod version.
    :param framework_info_file Path to the framework_info.json file containing additional values for the podspec.
    :param package_variant The pod package variant.
    :return Tuple of (package name, path to the podspec file).
    """
    staging_dir = staging_dir.resolve()
    framework_info_file = framework_info_file.resolve(strict=True)

    framework_info = load_json_config(framework_info_file)
    pod_config = load_json_config(get_pod_config_file(package_variant))
    c_pod_config = load_json_config(get_c_pod_config_file(package_variant))

    pod_name = pod_config["name"]

    print(f"Assembling files in staging directory: {staging_dir}")
    if staging_dir.exists():
        print("Warning: staging directory already exists", file=sys.stderr)

    pod_files = get_pod_files(package_variant)

    # copy the necessary files to the staging directory
    copy_repo_relative_to_dir(
        [license_file, *pod_files["source_files"], *pod_files["test_source_files"], *pod_files["test_resource_files"]],
        staging_dir,
    )

    # generate the podspec file from the template

    def path_patterns_as_variable_value(patterns: list[str]):
        return ", ".join([f'"{pattern}"' for pattern in patterns])

    variable_substitutions = {
        "C_POD_NAME": c_pod_config["name"],
        "DESCRIPTION": pod_config["description"],
        "INCLUDE_DIR_LIST": path_patterns_as_variable_value(include_dirs),
        "IOS_DEPLOYMENT_TARGET": framework_info["IOS_DEPLOYMENT_TARGET"],
        "LICENSE_FILE": license_file,
        "NAME": pod_name,
        "PUBLIC_HEADER_FILE_LIST": path_patterns_as_variable_value(pod_files["public_header_files"]),
        "SOURCE_FILE_LIST": path_patterns_as_variable_value(pod_files["source_files"]),
        "SUMMARY": pod_config["summary"],
        "TEST_RESOURCE_FILE_LIST": path_patterns_as_variable_value(pod_files["test_resource_files"]),
        "TEST_SOURCE_FILE_LIST": path_patterns_as_variable_value(pod_files["test_source_files"]),
        "VERSION": pod_version,
    }

    podspec_template = _script_dir / "objc.podspec.template"
    podspec = staging_dir / f"{pod_name}.podspec"

    gen_file_from_template(podspec_template, podspec, variable_substitutions)

    return pod_name, podspec


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Assembles the files for the Objective-C pod package in a staging directory.
        This directory can be validated (e.g., with `pod lib lint`) and then zipped to create a package for release.
    """
    )

    parser.add_argument(
        "--staging-dir",
        type=pathlib.Path,
        default=pathlib.Path("./onnxruntime-mobile-objc-staging"),
        help="Path to the staging directory for the Objective-C pod files.",
    )
    parser.add_argument("--pod-version", required=True, help="Objective-C pod version.")
    parser.add_argument(
        "--framework-info-file",
        type=pathlib.Path,
        required=True,
        help="Path to the framework_info.json file containing additional values for the podspec. "
        "This file should be generated by CMake in the build directory.",
    )
    parser.add_argument(
        "--variant", choices=PackageVariant.release_variant_names(), required=True, help="Pod package variant."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assemble_objc_pod_package(
        staging_dir=args.staging_dir,
        pod_version=args.pod_version,
        framework_info_file=args.framework_info_file,
        package_variant=PackageVariant[args.variant],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
