#!/bin/bash
# Install packages that will be needed at runtime

# Stop at any error, show all commands
set -exuo pipefail


# if a devel package is added to COMPILE_DEPS,
# make sure the corresponding library is added to RUNTIME_DEPS if applicable

if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ] || [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
	PACKAGE_MANAGER=yum
	COMPILE_DEPS="zlib-devel bzip2-devel expat-devel ncurses-devel readline-devel tk-devel gdbm-devel libpcap-devel xz-devel openssl openssl-devel keyutils-libs-devel krb5-devel libcom_err-devel libidn-devel curl-devel uuid-devel libffi-devel kernel-headers"
	if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ]; then
		COMPILE_DEPS="${COMPILE_DEPS} db4-devel"
	else
		COMPILE_DEPS="${COMPILE_DEPS} libdb-devel"
	fi
elif [ "${AUDITWHEEL_POLICY}" == "manylinux_2_24" ]; then
	PACKAGE_MANAGER=apt
	COMPILE_DEPS="libz-dev libbz2-dev libexpat1-dev libncurses5-dev libreadline-dev tk-dev libgdbm-dev libdb-dev libpcap-dev liblzma-dev openssl libssl-dev libkeyutils-dev libkrb5-dev comerr-dev libidn2-0-dev libcurl4-openssl-dev uuid-dev libffi-dev linux-kernel-headers"
else
	echo "Unsupported policy: '${AUDITWHEEL_POLICY}'"
	exit 1
fi


if [ "${PACKAGE_MANAGER}" == "yum" ]; then
	yum -y install ${COMPILE_DEPS}
	yum clean all
	rm -rf /var/cache/yum
elif [ "${PACKAGE_MANAGER}" == "apt" ]; then
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -qq
	apt-get install -qq -y --no-install-recommends ${COMPILE_DEPS}
	apt-get clean -qq
	rm -rf /var/lib/apt/lists/*
else
	echo "Not implemented"
	exit 1
fi
