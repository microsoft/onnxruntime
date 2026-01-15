#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

runner_home=/home/ortqnnepci
runner_root="${runner_home}/actions-runner"

if [ ! -d "${runner_root}" ]; then
    echo "Install (but do not enable) the GitHub Actions runner before running this script."
    echo "Be sure to run everything as user ortqnnepci (e.g., sudo -u ortqnnepci mkdir actions-runner)."
    echo "See https://github.qualcomm.com/MLG/onnxruntime-qnn-ep/settings/actions/runners"
    exit 1
fi

set -x

sudo add-apt-repository --yes ppa:deadsnakes/ppa

sudo apt update

sudo apt --yes install \
    net-tools \
    python3.10-dev python3.10-venv \
    python3.11-dev python3.11-venv \
    python3.12-dev python3.12-venv

##########################################
# Configure runner's environment variables
env_tmpfile=$(mktemp --suffix=-runner-env)
cat > "${env_tmpfile}" <<EOF
LANG=en_US.UTF-8
HOME=${runner_home}

ORT_BUILD_PACKAGE_CACHE_PATH=${runner_home}/ort-package-cache
ORT_BUILD_TOOLS_PATH=${runner_home}/ort-build-tools
EOF

chmod 644 "${env_tmpfile}"
sudo -u ortqnnepci cp "${env_tmpfile}" "${runner_root}/.env"
rm "${env_tmpfile}"

set +x
echo
echo "-=-=-=-=-=- Setup complete -=-=-=-=-=-"
echo
echo "Next steps:"
echo "  cd ${runner_root}"
echo "  sudo ./svc.sh install ortqnnepci"
echo "  sudo ./svc.sh start"
