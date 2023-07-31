#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for os major version : $os_major_version"
if [ "$os_major_version" -gt 7 ]; then
    PACKAGE_MANAGER="dnf"
    $PACKAGE_MANAGER install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel perl-IPC-Cmd openssl-devel wget
else
    PACKAGE_MANAGER="yum"
    $PACKAGE_MANAGER install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind bzip2 bzip2-devel perl-IPC-Cmd openssl-devel wget
fi

# Install Java
# Install automatic documentation generation dependencies
$PACKAGE_MANAGER install -y java-11-openjdk-devel graphviz
