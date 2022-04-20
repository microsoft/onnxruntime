#!/bin/bash

# Stop at any error, show all commands
set -exuo pipefail

# Get script directory
MY_DIR=$(dirname "${BASH_SOURCE[0]}")

# Get build utilities
source $MY_DIR/build_utils.sh

mkdir /opt/python
for PREFIX in $(find /opt/_internal/ -mindepth 1 -maxdepth 1 \( -name 'cpython*' -o -name 'pypy*' \)); do
	# Some python's install as bin/python3. Make them available as
	# bin/python.
	if [ -e ${PREFIX}/bin/python3 ] && [ ! -e ${PREFIX}/bin/python ]; then
		ln -s python3 ${PREFIX}/bin/python
	fi
	${PREFIX}/bin/python -m ensurepip
	if [ -e ${PREFIX}/bin/pip3 ] && [ ! -e ${PREFIX}/bin/pip ]; then
		ln -s pip3 ${PREFIX}/bin/pip
	fi
	PY_VER=$(${PREFIX}/bin/python -c "import sys; print('.'.join(str(v) for v in sys.version_info[:2]))")
	# Since we fall back on a canned copy of pip, we might not have
	# the latest pip and friends. Upgrade them to make sure.
	${PREFIX}/bin/pip install -U --require-hashes -r ${MY_DIR}/requirements${PY_VER}.txt
	# Create a symlink to PREFIX using the ABI_TAG in /opt/python/
	ABI_TAG=$(${PREFIX}/bin/python ${MY_DIR}/python-tag-abi-tag.py)
	ln -s ${PREFIX} /opt/python/${ABI_TAG}
	# Make versioned python commands available directly in environment.
	if [[ "${PREFIX}" == *"/pypy"* ]]; then
		ln -s ${PREFIX}/bin/python /usr/local/bin/pypy${PY_VER}
	else
		ln -s ${PREFIX}/bin/python /usr/local/bin/python${PY_VER}
	fi
done

# Create venv for auditwheel & certifi
TOOLS_PATH=/opt/_internal/tools
/opt/python/cp39-cp39/bin/python -m venv $TOOLS_PATH
source $TOOLS_PATH/bin/activate

# Install default packages
pip install -U --require-hashes -r $MY_DIR/requirements3.9.txt
# Install certifi and pipx
pip install -U --require-hashes -r $MY_DIR/requirements-base-tools.txt

# Make pipx available in PATH,
# Make sure when root installs apps, they're also in the PATH
cat <<EOF > /usr/local/bin/pipx
#!/bin/bash

set -euo pipefail

if [ \$(id -u) -eq 0 ]; then
	export PIPX_HOME=/opt/_internal/pipx
	export PIPX_BIN_DIR=/usr/local/bin
fi
${TOOLS_PATH}/bin/pipx "\$@"
EOF
chmod 755 /usr/local/bin/pipx

# Our openssl doesn't know how to find the system CA trust store
#   (https://github.com/pypa/manylinux/issues/53)
# And it's not clear how up-to-date that is anyway
# So let's just use the same one pip and everyone uses
ln -s $(python -c 'import certifi; print(certifi.where())') /opt/_internal/certs.pem
# If you modify this line you also have to modify the versions in the Dockerfiles:
export SSL_CERT_FILE=/opt/_internal/certs.pem

# Deactivate the tools virtual environment
deactivate

# install other tools with pipx
pushd $MY_DIR/requirements-tools
for TOOL_PATH in $(find . -type f); do
	TOOL=$(basename ${TOOL_PATH})
	pipx install --pip-args="--require-hashes -r" ${TOOL}
done
popd

# We do not need the precompiled .pyc and .pyo files.
clean_pyc /opt/_internal

# remove cache
rm -rf /root/.cache

hardlink -cv /opt/_internal

# update system packages
LC_ALL=C ${MY_DIR}/update-system-packages.sh
