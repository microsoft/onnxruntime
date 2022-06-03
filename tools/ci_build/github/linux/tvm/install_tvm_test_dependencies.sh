#!/bin/bash
set -e -x

PYTHON_EXE=$1
${PYTHON_EXE} -m pip install decorator scipy

sudo apt-get update && sudo apt-get install -y libtinfo5
