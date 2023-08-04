# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This script is used to add trigger rules to the workflow files.
# The working directory should be in tools/ci_build/github/azure-pipelines

import multiprocessing


def add_trigger_filter(file_name, trigger_lines):
    # Open the file and read its lines
    with open(file_name) as f:
        lines = f.readlines()

    start_marker = "##### trigger Don't modified it manully ####"
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

def main():
    workflow_files = ["linux-gpu-ci-pipeline.yml", "win-gpu-ci-pipeline.yml"]

    trigger_file = "trigger-template.yml"
    with open(trigger_file) as f1:
        trigger_lines = f1.readlines()

    pool = multiprocessing.Pool()
    pool.starmap(add_trigger_filter, [(file, trigger_lines) for file in workflow_files])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
