#!/bin/bash
# Install packages that will be needed at runtime

# Stop at any error, show all commands
set -exuo pipefail

# Set build environment variables
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Libraries that are allowed as part of the manylinux2014 profile
# Extract from PEP: https://www.python.org/dev/peps/pep-0599/#the-manylinux2014-policy
# On RPM-based systems, they are provided by these packages:
# Package:    Libraries
# glib2:      libglib-2.0.so.0, libgthread-2.0.so.0, libgobject-2.0.so.0
# glibc:      libresolv.so.2, libutil.so.1, libnsl.so.1, librt.so.1, libpthread.so.0, libdl.so.2, libm.so.6, libc.so.6
# libICE:     libICE.so.6
# libX11:     libX11.so.6
# libXext:    libXext.so.6
# libXrender: libXrender.so.1
# libgcc:     libgcc_s.so.1
# libstdc++:  libstdc++.so.6
# mesa:       libGL.so.1
#
# PEP is missing the package for libSM.so.6 for RPM based system

# MANYLINUX_DEPS: Install development packages (except for libgcc which is provided by gcc install)
if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ] || [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
	MANYLINUX_DEPS="glibc-devel libstdc++-devel glib2-devel libX11-devel libXext-devel libXrender-devel mesa-libGL-devel libICE-devel libSM-devel"
elif [ "${AUDITWHEEL_POLICY}" == "manylinux_2_24" ]; then
	MANYLINUX_DEPS="libc6-dev libstdc++-6-dev libglib2.0-dev libx11-dev libxext-dev libxrender-dev libgl1-mesa-dev libice-dev libsm-dev"
else
	echo "Unsupported policy: '${AUDITWHEEL_POLICY}'"
	exit 1
fi

# RUNTIME_DEPS: Runtime dependencies. c.f. install-build-packages.sh
if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ] || [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
	RUNTIME_DEPS="zlib bzip2 expat ncurses readline tk gdbm libpcap xz openssl keyutils-libs libkadm5 libcom_err libidn libcurl uuid libffi"
	if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ]; then
		RUNTIME_DEPS="${RUNTIME_DEPS} db4"
	else
		RUNTIME_DEPS="${RUNTIME_DEPS} libdb"
	fi
elif [ "${AUDITWHEEL_POLICY}" == "manylinux_2_24" ]; then
	RUNTIME_DEPS="zlib1g libbz2-1.0 libexpat1 libncurses5 libreadline7 tk libgdbm3 libdb5.3 libpcap0.8 liblzma5 libssl1.1 libkeyutils1 libkrb5-3 libcomerr2 libidn2-0 libcurl3 uuid libffi6"
else
	echo "Unsupported policy: '${AUDITWHEEL_POLICY}'"
	exit 1
fi

BASETOOLS="autoconf automake bison bzip2 diffutils file hardlink make patch unzip"
if [ "${AUDITWHEEL_POLICY}" == "manylinux2010" ]; then
	PACKAGE_MANAGER=yum
	BASETOOLS="${BASETOOLS} which"
	# See https://unix.stackexchange.com/questions/41784/can-yum-express-a-preference-for-x86-64-over-i386-packages
	echo "multilib_policy=best" >> /etc/yum.conf
	fixup-mirrors
	yum -y update
	fixup-mirrors
	yum -y install https://archives.fedoraproject.org/pub/archive/epel/6/x86_64/epel-release-6-8.noarch.rpm curl
	fixup-mirrors
	TOOLCHAIN_DEPS="devtoolset-8-binutils devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran yasm"
	if [ "${AUDITWHEEL_ARCH}" == "x86_64" ]; then
		# Software collection (for devtoolset-8)
		yum -y install centos-release-scl
		fixup-mirrors
	elif [ "${AUDITWHEEL_ARCH}" == "i686" ]; then
		# Add libgfortran4 for devtoolset-7 compat
		TOOLCHAIN_DEPS="${TOOLCHAIN_DEPS} libgfortran4"
		# Install mayeut/devtoolset-8 repo to get devtoolset-8
		curl -fsSLo /etc/yum.repos.d/mayeut-devtoolset-8.repo https://copr.fedorainfracloud.org/coprs/mayeut/devtoolset-8-i386/repo/custom-1/mayeut-devtoolset-8-i386-custom-1.repo
	fi
elif [ "${AUDITWHEEL_POLICY}" == "manylinux2014" ]; then
	PACKAGE_MANAGER=yum
	BASETOOLS="${BASETOOLS} hostname which"
	# See https://unix.stackexchange.com/questions/41784/can-yum-express-a-preference-for-x86-64-over-i386-packages
	echo "multilib_policy=best" >> /etc/yum.conf
	# Error out if requested packages do not exist
	echo "skip_missing_names_on_install=False" >> /etc/yum.conf
	# Make sure that locale will not be removed
	sed -i '/^override_install_langs=/d' /etc/yum.conf
	# Exclude mirror holding broken package metadata
	echo "exclude = d36uatko69830t.cloudfront.net" >> /etc/yum/pluginconf.d/fastestmirror.conf
	yum -y update
	yum -y install yum-utils curl
	yum-config-manager --enable extras
	#Added by @snnn
	if [ ! -d "/usr/local/cuda-10.2" ]; then
	  TOOLCHAIN_DEPS="devtoolset-9-binutils devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-gcc-gfortran"
	else
	  TOOLCHAIN_DEPS="devtoolset-8-binutils devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran"
	fi
	if [ "${AUDITWHEEL_ARCH}" == "x86_64" ]; then
		# Software collection (for devtoolset-9)
		yum -y install centos-release-scl-rh
		# EPEL support (for yasm)
		if ! rpm -q --quiet epel-release ; then
		  yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
		fi
		TOOLCHAIN_DEPS="${TOOLCHAIN_DEPS} yasm"
	elif [ "${AUDITWHEEL_ARCH}" == "aarch64" ] || [ "${AUDITWHEEL_ARCH}" == "ppc64le" ] || [ "${AUDITWHEEL_ARCH}" == "s390x" ]; then
		# Software collection (for devtoolset-9)
		yum -y install centos-release-scl-rh
	elif [ "${AUDITWHEEL_ARCH}" == "i686" ]; then
		# No yasm on i686
		# Install mayeut/devtoolset-9 repo to get devtoolset-9
		curl -fsSLo /etc/yum.repos.d/mayeut-devtoolset-9.repo https://copr.fedorainfracloud.org/coprs/mayeut/devtoolset-9/repo/custom-1/mayeut-devtoolset-9-custom-1.repo
	fi
elif [ "${AUDITWHEEL_POLICY}" == "manylinux_2_24" ]; then
	PACKAGE_MANAGER=apt
	BASETOOLS="${BASETOOLS} hostname"
	export DEBIAN_FRONTEND=noninteractive
	sed -i 's/none/en_US/g' /etc/apt/apt.conf.d/docker-no-languages
	apt-get update -qq
	apt-get upgrade -qq -y
	apt-get install -qq -y --no-install-recommends ca-certificates gpg curl locales
	TOOLCHAIN_DEPS="binutils gcc g++ gfortran"
else
	echo "Unsupported policy: '${AUDITWHEEL_POLICY}'"
	exit 1
fi

if [ "${PACKAGE_MANAGER}" == "yum" ]; then
	yum -y install ${BASETOOLS} ${TOOLCHAIN_DEPS} ${MANYLINUX_DEPS} ${RUNTIME_DEPS}
elif [ "${PACKAGE_MANAGER}" == "apt" ]; then
	apt-get install -qq -y --no-install-recommends ${BASETOOLS} ${TOOLCHAIN_DEPS} ${MANYLINUX_DEPS} ${RUNTIME_DEPS}
else
	echo "Not implemented"
	exit 1
fi

# update system packages, we already updated them but
# the following script takes care of cleaning-up some things
# and since it's also needed in the finalize step, everything's
# centralized in this script to avoid code duplication
LC_ALL=C ${MY_DIR}/update-system-packages.sh

# we'll be removing libcrypt.so.1 later on
# this is needed to ensure the new one will be found
# as LD_LIBRARY_PATH does not seem enough.
# c.f. https://github.com/pypa/manylinux/issues/1022
echo "/usr/local/lib" > /etc/ld.so.conf.d/00-manylinux.conf
ldconfig
