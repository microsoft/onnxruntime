#!/bin/sh
# Install entrypoint:
#   - make sure yum is configured correctly and linux32 is available on i686
#   - install bash on Alpine as most scripts require it

# Stop at any error, show all commands
set -exu

# Set build environment variables
MY_DIR=$(dirname "$0")


if [ "${AUDITWHEEL_PLAT}" = "manylinux2010_i686" ] || [ "${AUDITWHEEL_PLAT}" = "manylinux2014_i686" ]; then
	echo "i386" > /etc/yum/vars/basearch
	fixup-mirrors
	yum -y update
	fixup-mirrors
	yum install -y util-linux-ng
	# update system packages, we already updated them but
	# the following script takes care of cleaning-up some things
	# and since it's also needed in the finalize step, everything's
	# centralized in this script to avoid code duplication
	LC_ALL=C "${MY_DIR}/update-system-packages.sh"
fi

if [ "${AUDITWHEEL_POLICY}" = "musllinux_1_1" ]; then
	apk add --no-cache bash
fi

if [ command -v yum &> /dev/null ];
then
	yum install -y yum-plugin-versionlock
	yum versionlock cuda* libcudnn* libnccl*
fi
