# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This script is used to add trigger rules to the workflow files.


import multiprocessing
import os
from os.path import abspath, dirname

skip_doc_changes = ["web-ci-pipeline.yml"]
skip_js_changes = [
    "android-arm64-v8a-QNN-crosscompile-ci-pipeline.yml",
    "android-x86_64-crosscompile-ci-pipeline.yml",
    "linux-ci-pipeline.yml",
    "linux-cpu-aten-pipeline.yml",
    "linux-cpu-eager-pipeline.yml",
    "linux-dnnl-ci-pipeline.yml",
    "linux-gpu-ci-pipeline.yml",
    "linux-gpu-tensorrt-ci-pipeline.yml",
    "linux-migraphx-ci-pipeline.yml",
    "linux-openvino-ci-pipeline.yml",
    "linux-qnn-ci-pipeline.yml",
    "mac-ci-pipeline.yml",
    "mac-coreml-ci-pipeline.yml",
    "mac-ios-ci-pipeline.yml",
    "mac-ios-packaging-pipeline.yml",
    "mac-react-native-ci-pipeline.yml",
    "orttraining-linux-ci-pipeline.yml",
    "orttraining-linux-gpu-ci-pipeline.yml",
    "orttraining-linux-gpu-ortmodule-distributed-test-ci-pipeline.yml",
    "orttraining-linux-gpu-training-apis.yml",
    "orttraining-mac-ci-pipeline.yml",
    "win-ci-pipeline.yml",
    "win-gpu-ci-pipeline.yml",
    "win-gpu-tensorrt-ci-pipeline.yml",
    "win-qnn-arm64-ci-pipeline.yml",
    "win-qnn-ci-pipeline.yml",
]


def add_trigger_filter(file_name, trigger_lines):
    # Open the file and read its lines
    with open(file_name) as f:
        lines = f.readlines()

    start_marker = f"##### start trigger Don't edit it manually, Please do edit {os.path.basename(__file__)} ####"
    end_marker = "#### end trigger ####\n"

    if lines[0].startswith(start_marker):
        for i in range(1, len(lines)):
            if lines[i].startswith(end_marker):
                lines[1:i] = trigger_lines
                break
    else:
        trigger_lines.insert(0, start_marker + "\n")
        trigger_lines.extend(end_marker + "\n")
        lines[0:0] = trigger_lines

    with open(file_name, "w") as f:
        f.writelines(lines)
        print("Added trigger rules to file: " + file_name)


def main():
    working_dir = os.path.join(dirname(abspath(__file__)), "github/azure-pipelines")
    os.chdir(working_dir)

    trigger_rules = {"skip-docs.yml": skip_doc_changes, "skip-js.yml": skip_js_changes}
    for key in trigger_rules:
        trigger_file = os.path.join(working_dir, "triggers", key)
        with open(trigger_file) as f1:
            trigger_lines = f1.readlines()

        skip_changes = trigger_rules[key]
        pool = multiprocessing.Pool()
        pool.starmap(add_trigger_filter, [(file, trigger_lines) for file in skip_changes])
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
