#!/bin/bash
# Fix up mirrors once distro reaches EOL

# Stop at any error, show all commands
set -exuo pipefail

if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ]; then
	# Centos 6 is EOL and is no longer available from the usual mirrors, so switch
	# to https://vault.centos.org
	sed -i 's/enabled=1/enabled=0/g' /etc/yum/pluginconf.d/fastestmirror.conf
	sed -i 's/^mirrorlist/#mirrorlist/g' /etc/yum.repos.d/*.repo
	sed -i 's;^.*baseurl=http://mirror;baseurl=https://vault;g' /etc/yum.repos.d/*.repo
	sed -i 's;^.*baseurl=http://download.fedoraproject.org/pub;baseurl=https://archives.fedoraproject.org/pub/archive;g' /etc/yum.repos.d/*.repo
fi
