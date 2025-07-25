#!/usr/bin/env bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

set -euo pipefail

runner_root=/local/mnt/workspace/actions-runner
runner_home="${runner_root}/_ort-cache"

if [ ! -d "${runner_root}" ]; then
    echo "Install (but do not enable) the GitHub Actions runner before running this script."
    echo "Be sure to run everything as user ortqnnepci (e.g., sudo -u ortqnnepci mkdir actions-runner)."
    echo "See https://github.qualcomm.com/MLG/onnxruntime-qnn-ep/settings/actions/runners"
    exit 1
fi

set -x

sudo apt install libc++-dev python3.10-dev python3.10-venv

# Configure GPG keyrings and other dependencies for GitHub CLI
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

sudo -u ortqnnepci mkdir -p "${runner_home}"

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

##################
# Configure ccache
ccache_tmpfile=$(mktemp --suffix=-ccache-conf)
cat > "${ccache_tmpfile}" <<EOF
max_size = 10G
EOF

chmod 644 "${ccache_tmpfile}"
sudo -u ortqnnepci mkdir -p "${runner_home}/.ccache"
sudo -u ortqnnepci cp "${ccache_tmpfile}" "${runner_home}/.ccache/ccache.conf"
rm "${ccache_tmpfile}"

set +x
echo
echo "-=-=-=-=-=- Setup complete -=-=-=-=-=-"
echo
echo "Next steps:"
echo "  cd ${runner_root}"
echo "  sudo ./svc.sh install ortqnnepci"
echo "  sudo ./svc.sh start"
