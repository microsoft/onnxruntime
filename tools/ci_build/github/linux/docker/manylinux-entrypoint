#!/bin/bash

set -eu

if [ "${AUDITWHEEL_ARCH}" == "i686" ]; then
	linux32 "$@"
else
	exec "$@"
fi
