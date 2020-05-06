#!/bin/bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Creates and activates a temporary conda environment and runs the specified command.
# Usage example:
#   $ PYTHON_VERSION=3.7 CONDA_INSTALL_DIR=/opt/miniconda3 run_in_temp_conda_env.sh python --version

PYTHON_VERSION=${PYTHON_VERSION:-3.6}
CONDA_INSTALL_DIR=${CONDA_INSTALL_DIR:?"Please specify the conda install directory in \$CONDA_INSTALL_DIR."}

# enable conda commands from bash script
source ${CONDA_INSTALL_DIR}/etc/profile.d/conda.sh || exit 1

function clean_up_and_exit() {
  local env_dir=$1
  local exit_code=$2

  if [ "${CONDA_DEFAULT_ENV}" == "${env_dir}" ]; then
    conda deactivate
  fi

  if [ -d "${env_dir}" ]; then
    conda env remove --yes --quiet --prefix ${env_dir}
    rm -rf ${env_dir}
  fi

  exit ${exit_code}
}

ENV_DIR=$(mktemp -d)
conda create --yes --quiet --prefix ${ENV_DIR} python=${PYTHON_VERSION} || clean_up_and_exit ${ENV_DIR} 1
conda activate ${ENV_DIR} || clean_up_and_exit ${ENV_DIR} 1

$@

clean_up_and_exit ${ENV_DIR} $?
