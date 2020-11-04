#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import argparse
import subprocess

import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)

log = logging.getLogger("Build")

def run_subprocess(args, cwd=None, env={}):
    log.info("Running subprocess in '{0}'\n{1}".format(cwd or os.getcwd(), args))
    my_env = os.environ.copy()
    my_env.update(env)
    completed_process = subprocess.run(args, cwd=cwd, check=True, env=my_env)
    log.debug("Subprocess completed. Return code=" +
              str(completed_process.returncode))
    return completed_process

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_args", required=True, help="process args.")
    parser.add_argument("--cwd", help="cwd")
    parser.add_argument("--dll_path", help="dll path.")
    parser.add_argument("--env", help="env variables.")
    return parser.parse_args()

def list_to_dict(list_key_values):
    assert len(list_key_values) % 2 == 0, "list should have even length"
    my_dictionary = {}
    for i in range(0, len(list_key_values), 2):
        my_dictionary[list_key_values[i]] = list_key_values[i + 1]

    return my_dictionary

launch_args = parse_arguments()

print("sys.executable: ", sys.executable)
process_args = launch_args.process_args.split()
for n, arg in enumerate(process_args):
    if arg == 'python':
        process_args[n] = sys.executable

cwd = launch_args.cwd

env = list_to_dict(launch_args.env.split()) if launch_args.env else {}

run_subprocess(process_args, cwd=cwd, env=env)
