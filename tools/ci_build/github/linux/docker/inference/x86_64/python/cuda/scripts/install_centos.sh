#!/bin/bash
set -e

os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)

echo "installing for os major version : $os_major_version"
dnf install -y glibc-langpack-\* which redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel perl-IPC-Cmd openssl-devel wget
