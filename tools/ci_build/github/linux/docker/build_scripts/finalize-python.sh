#!/bin/bash

# Stop at any error, show all commands
set -exuo pipefail

# most people don't need libpython*.a, and they're many megabytes.
# compress them all together for best efficiency
pushd /opt/_internal
XZ_OPT=-9e tar -cJf static-libs-for-embedding-only.tar.xz cpython-*/lib/libpython*.a
popd
find /opt/_internal -name '*.a' -print0 | xargs -0 rm -f

hardlink -cv /opt/_internal
