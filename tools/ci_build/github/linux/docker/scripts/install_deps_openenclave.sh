#!/bin/bash
set -e

OPENENCLAVE_VERSION=0.9.0

echo "Installing common tools"
# gawk: used by cmake/external/openenclave_get_c_compiler_inc_dir.sh
apt-get update && apt-get -y --no-install-recommends install \
    gawk

echo "Installing Intel SGX libraries"

# Work-around for https://github.com/intel/linux-sgx/issues/395
mkdir -p /etc/init

curl https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -
apt-add-repository -y https://download.01.org/intel-sgx/sgx_repo/ubuntu
apt-get update && apt-get -y --no-install-recommends install \
    libsgx-enclave-common \
    libsgx-enclave-common-dev \
    libsgx-dcap-ql \
    libsgx-dcap-ql-dev

echo "Installing Open Enclave $OPENENCLAVE_VERSION"
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
apt-add-repository -y https://packages.microsoft.com/ubuntu/$(lsb_release -s -r)/prod
apt-get update && apt-get -y --no-install-recommends install \
    open-enclave=$OPENENCLAVE_VERSION