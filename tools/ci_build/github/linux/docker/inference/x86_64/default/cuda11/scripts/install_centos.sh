#!/bin/bash
set -e -x
if [ ! -f /etc/yum.repos.d/microsoft-prod.repo ]; then
  os_major_version=$(tr -dc '0-9.' < /etc/redhat-release |cut -d \. -f1)
  echo "installing for CentOS version : $os_major_version"
  rpm -Uvh https://packages.microsoft.com/config/centos/$os_major_version/packages-microsoft-prod.rpm
fi
dnf install -y python39-devel glibc-langpack-\* glibc-locale-source which redhat-lsb-core expat-devel tar unzip zlib-devel make bzip2 bzip2-devel msopenjdk-11
locale
