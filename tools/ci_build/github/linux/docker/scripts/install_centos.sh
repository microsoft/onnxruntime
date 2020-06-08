#!/bin/bash
set -e

os_major_version=$(cat /etc/redhat-release | tr -dc '0-9.'|cut -d \. -f1)


if ! rpm -q --quiet epel-release ; then
  yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-$os_major_version.noarch.rpm
fi

echo "installing for os major version : $os_major_version"
if [ "$os_major_version" == "5" ]; then
  #Be careful, don't pull gcc into the base system, because we already have one in /opt/rh/devtoolset-2/root/usr/bin
  yum install -y redhat-lsb expat-devel libcurl-devel tar unzip curl zlib-devel make icu  rsync bzip2 git bzip2-devel
elif [ "$os_major_version" == "6" ] && [ ! -d "/opt/python/cp35-cp35m" ]; then
  #The base image we are using already contains devtoolset-2
  yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make libunwind icu aria2 rsync bzip2 git bzip2-devel
  #Install python 3.6
  yum install -y https://repo.ius.io/ius-release-el6.rpm
  yum --enablerepo=ius install -y python36u python36u-devel python36u-pip python36u-numpy python36u-setuptools python36u-wheel
  /usr/bin/python3.6 -m pip install --upgrade pip
else
  yum install -y redhat-lsb-core expat-devel libcurl-devel tar unzip curl zlib-devel make libunwind icu aria2 rsync bzip2 git bzip2-devel

  if [ "$os_major_version" == "7" ]; then
    # install dotnet core dependencies
    yum install -y lttng-ust openssl-libs krb5-libs libicu libuuid
    # install dotnet runtimes
    yum install -y https://packages.microsoft.com/config/centos/7/packages-microsoft-prod.rpm
    yum install -y dotnet-sdk-2.2
  fi
fi

yum install -y java-1.8.0-openjdk-devel

#If the /opt/python folder exists, we assume this is the manylinux docker image
if [ "$os_major_version" != "6" ] && [ ! -d "/opt/python/cp35-cp35m"  ] 
then
  yum install -y ccache gcc gcc-c++ python3 python3-devel python3-pip
fi
