# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os
import shutil

from generate_nuspec_for_native_nuget import generate_metadata


def generate_files(lines, args):
    files_list = ["<files>"]
    platform_map = {
        "win-arm64": args.win_arm64,
        "win-x64": args.win_x64,
    }

    avoid_keywords = {"pdb"}
    processed_includes = set()
    for platform, platform_dir in platform_map.items():
        for file in glob.glob(os.path.join(platform_dir, "lib", "*")):
            if not os.path.isfile(file):
                continue
            if any(keyword in file for keyword in avoid_keywords):
                continue
            file_name = os.path.basename(file)

            files_list.append(f'<file src="{file}" target="runtimes/{platform}/native/{file_name}" />')

        for file in glob.glob(os.path.join(platform_dir, "include", "*")):
            if not os.path.isfile(file):
                continue
            file_name = os.path.basename(file)
            if file_name in processed_includes:
                continue
            processed_includes.add(file_name)
            files_list.append(f'<file src="{file}" target="build/native/include/{file_name}" />')

    files_list.append(
        f'<file src="{os.path.join(args.root_dir, "tools", "nuget", "nupkg.README.md")}" target="README.md" />'
    )

    files_list.append(f'<file src="{os.path.join(args.root_dir, "LICENSE")}" target="LICENSE" />')
    files_list.append(
        f'<file src="{os.path.join(args.root_dir, "ThirdPartyNotices.txt")}" target="ThirdPartyNotices.txt" />'
    )
    files_list.append(f'<file src="{os.path.join(args.root_dir, "docs", "Privacy.md")}" target="Privacy.md" />')
    files_list.append(
        f'<file src="{os.path.join(args.root_dir, "ORT_icon_for_light_bg.png")}" target="ORT_icon_for_light_bg.png" />'
    )

    source_props = os.path.join(
        args.root_dir,
        "csharp",
        "src",
        "Microsoft.ML.OnnxRuntime",
        "targets",
        "netstandard",
        "props.xml",
    )
    target_props = os.path.join(
        args.root_dir,
        "csharp",
        "src",
        "Microsoft.ML.OnnxRuntime",
        "targets",
        "netstandard",
        f"{args.package_name}.props",
    )
    shutil.copyfile(source_props, target_props)
    files_list.append(f'<file src="{target_props}" target="build/netstandard2.0/" />')
    files_list.append(f'<file src="{target_props}" target="build/netstandard2.1/" />')

    source_targets = os.path.join(
        args.root_dir,
        "csharp",
        "src",
        "Microsoft.ML.OnnxRuntime",
        "targets",
        "netstandard",
        "targets.xml",
    )
    target_targets = os.path.join(
        args.root_dir,
        "csharp",
        "src",
        "Microsoft.ML.OnnxRuntime",
        "targets",
        "netstandard",
        f"{args.package_name}.targets",
    )
    shutil.copyfile(source_targets, target_targets)
    files_list.append(f'<file src="{target_targets}" target="build/netstandard2.0/" />')
    files_list.append(f'<file src="{target_targets}" target="build/netstandard2.1/" />')

    files_list.append("</files>")
    lines.extend(files_list)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create a nuspec file for the custom nuget package.",
    )

    parser.add_argument("--nuspec_path", required=True, help="Nuspec output file path.")
    parser.add_argument("--root_dir", required=True, help="ORT repository root.")
    parser.add_argument(
        "--commit_id",
        required=True,
        help="The last commit id included in this package.",
    )
    parser.add_argument("--win_arm64", required=True, help="Ort win-arm64 directory")
    parser.add_argument("--win_x64", required=True, help="Ort win-x64 directory")
    parser.add_argument("--package_version", required=True, help="Version of the package")
    parser.add_argument("--package_name", required=True, help="Name of the package")

    args = parser.parse_args()

    args.sdk_info = ""

    return args


def generate_nuspec(args: argparse.Namespace):
    lines = ['<?xml version="1.0"?>']
    lines.append("<package>")

    generate_metadata(lines, args)
    generate_files(lines, args)

    lines.append("</package>")
    return lines


def main():
    args = parse_arguments()

    lines = generate_nuspec(args)

    with open(os.path.join(args.nuspec_path), "w") as f:
        for line in lines:
            # Uncomment the printing of the line if you need to debug what's produced on a CI machine
            print(line)
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    main()
