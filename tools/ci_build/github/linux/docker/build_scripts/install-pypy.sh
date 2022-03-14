#!/bin/bash

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

if [ "${BASE_POLICY}" == "musllinux" ]; then
	echo "Skip PyPy build on musllinux"
	exit 0
fi

PYTHON_VERSION=$1
PYPY_VERSION=$2
PYPY_DOWNLOAD_URL=https://downloads.python.org/pypy


function get_shortdir {
	local exe=$1
	$exe -c 'import sys; print("pypy%d.%d-%d.%d.%d" % (sys.version_info[:2]+sys.pypy_version_info[:3]))'
}


mkdir -p /tmp
cd /tmp

case ${AUDITWHEEL_ARCH} in
	x86_64) PYPY_ARCH=linux64;;
	i686) PYPY_ARCH=linux32;;
	aarch64) PYPY_ARCH=aarch64;;
	*) echo "No PyPy for ${AUDITWHEEL_ARCH}"; exit 0;;
esac

TARBALL=pypy${PYTHON_VERSION}-v${PYPY_VERSION}-${PYPY_ARCH}.tar.bz2
TMPDIR=/tmp/${TARBALL/.tar.bz2//}
PREFIX="/opt/_internal"

mkdir -p ${PREFIX}

fetch_source ${TARBALL} ${PYPY_DOWNLOAD_URL}

# We only want to check the current tarball sha256sum
grep " ${TARBALL}\$" ${MY_DIR}/pypy.sha256 > ${TARBALL}.sha256
# then check sha256 sum
sha256sum -c ${TARBALL}.sha256

tar -xf ${TARBALL}

# the new PyPy 3 distributions don't have pypy symlinks to pypy3
if [ ! -f "${TMPDIR}/bin/pypy" ]; then
	ln -s pypy3 ${TMPDIR}/bin/pypy
fi

# rename the directory to something shorter like pypy3.7-7.3.4
PREFIX=${PREFIX}/$(get_shortdir ${TMPDIR}/bin/pypy)
mv ${TMPDIR} ${PREFIX}

# add a generic "python" symlink
if [ ! -f "${PREFIX}/bin/python" ]; then
	ln -s pypy ${PREFIX}/bin/python
fi

# remove debug symbols
rm ${PREFIX}/bin/*.debug

# We do not need the Python test suites
find ${PREFIX} -depth \( -type d -a -name test -o -name tests \) | xargs rm -rf

# We do not need precompiled .pyc and .pyo files.
clean_pyc ${PREFIX}
