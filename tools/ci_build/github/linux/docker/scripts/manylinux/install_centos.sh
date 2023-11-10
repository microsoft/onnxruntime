#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)

echo "installing for os major version : $os_major_version"
yum install -y which gdb redhat-lsb-core expat-devel tar unzip zlib-devel make libunwind bzip2 bzip2-devel


# Install Java
# Install automatic documentation generation dependencies
yum install -y java-11-openjdk-devel graphviz
