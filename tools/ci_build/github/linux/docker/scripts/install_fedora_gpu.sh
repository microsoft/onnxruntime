#!/bin/bash
set -e
dnf install -y --best --allowerasing compat-openssl10 lttng-ust libcurl openssl-libs krb5-libs libicu tar unzip ccache curl gcc gcc-c++ zlib-devel make git python2-devel python3-devel python3-numpy libunwind icu aria2 rsync python3-setuptools python3-wheel bzip2
mkdir -p /tmp/azcopy
aria2c -q -d /tmp/azcopy -o azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
tar -xf /tmp/azcopy/azcopy.tar.gz --strip 1  -C /tmp/azcopy
/bin/cp /tmp/azcopy/azcopy /usr/bin/azcopy
rm -rf /tmp/azcopy
#install cuda 9.2 and cudnn 7
#azcopy --source https://lotus.blob.core.windows.net/cudnn/linux/cuda/9.2/v7.2.1.38.tgz --destination /tmp/cudnn-9.2-linux-x64-v7.2.1.38.tgz --source-key $AZURE_BLOB_KEY
aria2c -q -d /tmp 'http://developer.download.nvidia.com/compute/cuda/repos/fedora27/x86_64/cuda-repo-fedora27-10.0.130-1.x86_64.rpm'
dnf install -y /tmp/cuda-repo-fedora27-10.0.130-1.x86_64.rpm
dnf install -y cuda-10.0.130-1.x86_64
tar -zxvf cudnn-10.0-linux-x64-v7.4.2.24.tgz --strip 1 -C /usr/local
ldconfig -v /usr/local/cuda/lib64 /usr/local/lib64

