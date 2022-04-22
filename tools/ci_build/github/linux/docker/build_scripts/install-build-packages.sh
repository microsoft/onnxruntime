#!/bin/bash
# Install packages that will be needed at runtime

# Stop at any error, show all commands
set -exuo pipefail


# if a devel package is added to COMPILE_DEPS,
# make sure the corresponding library is added to RUNTIME_DEPS if applicable

if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ] || [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
	PACKAGE_MANAGER=yum
	COMPILE_DEPS="bzip2-devel ncurses-devel readline-devel tk-devel gdbm-devel libpcap-devel xz-devel openssl openssl-devel keyutils-libs-devel krb5-devel libcom_err-devel libidn-devel curl-devel uuid-devel libffi-devel kernel-headers"
	if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ]; then
		COMPILE_DEPS="${COMPILE_DEPS} db4-devel"
	else
		COMPILE_DEPS="${COMPILE_DEPS} libdb-devel"
	fi
elif [ "${AUDITWHEEL_POLICY}" == "manylinux_2_24" ]; then
	PACKAGE_MANAGER=apt
	COMPILE_DEPS="libbz2-dev libncurses5-dev libreadline-dev tk-dev libgdbm-dev libdb-dev libpcap-dev liblzma-dev openssl libssl-dev libkeyutils-dev libkrb5-dev comerr-dev libidn2-0-dev libcurl4-openssl-dev uuid-dev libffi-dev linux-kernel-headers"
elif [ "${AUDITWHEEL_POLICY}" == "musllinux_1_1" ]; then
	PACKAGE_MANAGER=apk
	COMPILE_DEPS="bzip2-dev ncurses-dev readline-dev tk-dev gdbm-dev libpcap-dev xz-dev openssl openssl-dev keyutils-dev krb5-dev libcom_err libidn-dev curl-dev util-linux-dev libffi-dev linux-headers"
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
elif [ "${PACKAGE_MANAGER}" == "apk" ]; then
	apk add --no-cache ${COMPILE_DEPS}
else
	echo "Not implemented"
	exit 1
fi
