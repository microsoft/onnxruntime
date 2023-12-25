#!/bin/bash
set -e -x

# version
ROCM_VERSION=6.0

while getopts "r:" parameter_Option
do case "${parameter_Option}"
in
r) ROCM_VERSION=${OPTARG};;
esac
done

tee /etc/yum.repos.d/amdgpu.repo <<EOF
[amdgpu]
name=amdgpu
baseurl=https://repo.radeon.com/amdgpu/$ROCM_VERSION/rhel/8.8/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
dnf clean all


# Enable epel-release repositories
# See: https://github.com/ROCm-Developer-Tools/HIP/issues/2330
dnf --enablerepo=extras install -y epel-release && dnf config-manager --set-enabled powertools

# Install the ROCm rpms
dnf clean all

tee /etc/yum.repos.d/rocm.repo <<EOF
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel8/$ROCM_VERSION/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

dnf install -y rocm-dev
