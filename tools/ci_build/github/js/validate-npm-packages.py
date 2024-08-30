#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys

# This is a script to validate NPM packages.
# If package version, publish tag and filename does not fulfill the requirement, an error will raise.

# arg.1 - Folder of extracted artifact "onnxruntime-node" for node.js binding
ort_node_pkg_dir = sys.argv[1]
# arg.2 - Folder of extracted artifact "onnxruntime-web" for web
ort_web_pkg_dir = sys.argv[2]
# arg.3 - Folder of extracted artifact "onnxruntime-react-native" for react native
ort_react_native_pkg_dir = sys.argv[3]
# arg.4 - source branch, eg. "refs/heads/master"
source_branch = sys.argv[4]
# arg.5 - NPM tag, eg. "", "dev", "latest", "rc"
tag = sys.argv[5]

# print out command line parameters
print("====== argv ======")
print("ort_node_pkg_dir:", ort_node_pkg_dir)
print("ort_web_pkg_dir:", ort_web_pkg_dir)
print("ort_react_native_pkg_dir:", ort_react_native_pkg_dir)
print("source_branch:", source_branch)
print("tag:", tag)

# check release flags from environment variables
RELEASE_NODE = os.environ.get("RELEASE_NODE", "") == "1"
RELEASE_WEB = os.environ.get("RELEASE_WEB", "") == "1"
RELEASE_REACT_NATIVE = os.environ.get("RELEASE_REACT_NATIVE", "") == "1"

# print ouf release flags
print("====== flags ======")
print("RELEASE_NODE:", RELEASE_NODE)
print("RELEASE_WEB:", RELEASE_WEB)
print("RELEASE_REACT_NATIVE:", RELEASE_REACT_NATIVE)

if not RELEASE_NODE and not RELEASE_WEB and not RELEASE_REACT_NATIVE:
    raise Exception("not releasing any package. exiting.")

count_ort_node_common_tgz = 0
count_ort_node_tgz = 0
ort_node_common_ver = ""
ort_node_ver = ""

for file in os.listdir(ort_node_pkg_dir):
    if file.startswith("onnxruntime-common-") and file.endswith(".tgz"):
        ort_node_common_ver = file[19:-4]
        count_ort_node_common_tgz += 1
    if file.startswith("onnxruntime-node-") and file.endswith(".tgz"):
        ort_node_ver = file[17:-4]
        count_ort_node_tgz += 1

count_ort_web_common_tgz = 0
count_ort_web_tgz = 0
ort_web_common_ver = ""
ort_web_ver = ""

for file in os.listdir(ort_web_pkg_dir):
    if file.startswith("onnxruntime-common-") and file.endswith(".tgz"):
        ort_web_common_ver = file[19:-4]
        count_ort_web_common_tgz += 1
    if file.startswith("onnxruntime-web-") and file.endswith(".tgz"):
        ort_web_ver = file[16:-4]
        count_ort_web_tgz += 1

count_ort_react_native_common_tgz = 0
count_ort_react_native_tgz = 0
ort_react_native_common_ver = ""
ort_react_native_ver = ""

for file in os.listdir(ort_react_native_pkg_dir):
    if file.startswith("onnxruntime-common-") and file.endswith(".tgz"):
        ort_react_native_common_ver = file[19:-4]
        count_ort_react_native_common_tgz += 1
    if file.startswith("onnxruntime-react-native-") and file.endswith(".tgz"):
        ort_react_native_ver = file[25:-4]
        count_ort_react_native_tgz += 1

if count_ort_node_common_tgz >= 2:
    raise Exception("expect at most 1 package file for onnxruntime-common in onnxruntime-node folder")
if count_ort_web_common_tgz >= 2:
    raise Exception("expect at most 1 package file for onnxruntime-common in onnxruntime-web folder")
if count_ort_react_native_common_tgz >= 2:
    raise Exception("expect at most 1 package file for onnxruntime-common in onnxruntime-react-native folder")

if RELEASE_NODE and RELEASE_WEB and count_ort_node_common_tgz != count_ort_web_common_tgz:
    raise Exception("inconsistent package number for onnxruntime-common (onnxruntime-node/onnxruntime-web)")
if RELEASE_NODE and RELEASE_REACT_NATIVE and count_ort_node_common_tgz != count_ort_react_native_common_tgz:
    raise Exception("inconsistent package number for onnxruntime-common (onnxruntime-node/onnxruntime-react-native)")
if RELEASE_WEB and RELEASE_REACT_NATIVE and count_ort_web_common_tgz != count_ort_react_native_common_tgz:
    raise Exception("inconsistent package number for onnxruntime-common (onnxruntime-web/onnxruntime-react-native)")

if RELEASE_NODE and RELEASE_WEB and ort_node_common_ver != ort_web_common_ver:
    raise Exception("inconsistent version number for onnxruntime-common (onnxruntime-node/onnxruntime-web)")
if RELEASE_NODE and RELEASE_REACT_NATIVE and ort_node_common_ver != ort_react_native_common_ver:
    raise Exception("inconsistent version number for onnxruntime-common (onnxruntime-node/onnxruntime-react-native)")
if RELEASE_WEB and RELEASE_REACT_NATIVE and ort_web_common_ver != ort_react_native_common_ver:
    raise Exception("inconsistent version number for onnxruntime-common (onnxruntime-web/onnxruntime-react-native)")

ort_common_ver = (
    ort_node_common_ver if RELEASE_NODE else (ort_web_common_ver if RELEASE_WEB else ort_react_native_common_ver)
)

ort_common_from = "" if not ort_common_ver else ("node" if RELEASE_NODE else ("web" if RELEASE_WEB else "react-native"))

print("====== output environment variables ======")
print(f"##vso[task.setvariable variable=ORT_COMMON_FROM]{ort_common_from}")

if tag == "latest" or tag == "" or tag == "rc":
    if not RELEASE_NODE or not RELEASE_WEB or not RELEASE_REACT_NATIVE:
        raise Exception("@latest or @rc build must release all packages (node, web, react-native)")
    if count_ort_node_common_tgz != 1:
        raise Exception("expect one package file for onnxruntime-common for release build")

if count_ort_node_tgz != 1:
    raise Exception("expect one package file for onnxruntime-node")
if count_ort_web_tgz != 1:
    raise Exception("expect one package file for onnxruntime-web")
if count_ort_react_native_tgz != 1:
    raise Exception("expect one package file for onnxruntime-react-native")
if RELEASE_NODE and RELEASE_WEB and ort_node_ver != ort_web_ver:
    raise Exception("version number is different for onnxruntime-node and onnxruntime-web")
if RELEASE_NODE and RELEASE_REACT_NATIVE and ort_node_ver != ort_react_native_ver:
    raise Exception("version number is different for onnxruntime-node and onnxruntime-react-native")
if RELEASE_WEB and RELEASE_REACT_NATIVE and ort_web_ver != ort_react_native_ver:
    raise Exception("version number is different for onnxruntime-web and onnxruntime-react-native")

print("====== validated versions ======")
print(f"source_branch={source_branch}")
print(f"tag={tag}")
print(f"ort_common_ver={ort_common_ver}")
print(f"ort_node_ver={ort_node_ver}")
print(f"ort_web_ver={ort_web_ver}")
print(f"ort_react_native_ver={ort_react_native_ver}")

if tag == "latest" or tag == "":
    print("Publishing @latest ...")
    if not source_branch.startswith("refs/heads/rel-"):
        raise Exception('@latest build must publish from source branch "refs/heads/rel-*"')
    if (
        "-" in ort_common_ver.replace("-rev", "")
        or "-" in ort_web_ver.replace("-rev", "")
        or "-" in ort_react_native_ver.replace("-rev", "")
    ):
        raise Exception('@latest build version cannot contain "-" (unless -rev)')
if tag == "rc":
    print("Publishing @rc ...")
    if not source_branch.startswith("refs/heads/rel-"):
        raise Exception('@rc build must publish from source branch "refs/heads/rel-*"')
    if "-rc" not in ort_web_ver:
        raise Exception('@rc build version should contain "-rc"')
    if "-rc" not in ort_react_native_ver:
        raise Exception('@rc build version should contain "-rc"')

if (
    "-" not in ort_common_ver.replace("-rev", "")
    and "-" not in ort_web_ver.replace("-rev", "")
    and "-" not in ort_react_native_ver.replace("-rev", "")
    and "+" not in ort_common_ver.replace("-rev", "")
    and "+" not in ort_web_ver.replace("-rev", "")
    and "+" not in ort_react_native_ver.replace("-rev", "")
):
    if tag != "latest" and tag != "":
        raise Exception("default version without decorator can only be published in @latest tag")
