#!/bin/bash
# Helper utilities for build


# use all flags used by ubuntu 20.04 for hardening builds, dpkg-buildflags --export
# other flags mentioned in https://wiki.ubuntu.com/ToolChain/CompilerFlags can't be
# used because the distros used here are too old
MANYLINUX_CPPFLAGS="-Wdate-time -D_FORTIFY_SOURCE=2"
MANYLINUX_CFLAGS="-g -O2 -Wall -fdebug-prefix-map=/=. -fstack-protector-strong -Wformat -Werror=format-security"
MANYLINUX_CXXFLAGS="-g -O2 -Wall -fdebug-prefix-map=/=. -fstack-protector-strong -Wformat -Werror=format-security"
MANYLINUX_LDFLAGS="-Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-z,now"


function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}


function fetch_source {
    # This is called both inside and outside the build context (e.g. in Travis) to prefetch
    # source tarballs, where curl exists (and works)
    local file=$1
    check_var ${file}
    local url=$2
    check_var ${url}
    if [ -f ${file} ]; then
        echo "${file} exists, skipping fetch"
    else
        curl -fsSL -o ${file} ${url}/${file}
    fi
}


function check_sha256sum {
    local fname=$1
    check_var ${fname}
    local sha256=$2
    check_var ${sha256}

    echo "${sha256}  ${fname}" > ${fname}.sha256
    sha256sum -c ${fname}.sha256
    rm -f ${fname}.sha256
}


function do_standard_install {
    ./configure "$@" CPPFLAGS="${MANYLINUX_CPPFLAGS}" CFLAGS="${MANYLINUX_CFLAGS}" "CXXFLAGS=${MANYLINUX_CXXFLAGS}" LDFLAGS="${MANYLINUX_LDFLAGS}" > /dev/null
    make > /dev/null
    make install > /dev/null
}

function strip_ {
	# Strip what we can -- and ignore errors, because this just attempts to strip
	# *everything*, including non-ELF files:
	find $1 -type f -print0 | xargs -0 -n1 strip --strip-unneeded 2>/dev/null || true
}

function clean_pyc {
	find $1 -type f -a \( -name '*.pyc' -o -name '*.pyo' \) -delete
}
