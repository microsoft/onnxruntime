# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

COLOR_GREEN='\033[0;32m'
COLOR_GREY='\033[0;37m'
COLOR_RED='\033[0;31m'
COLOR_RED_BOLD='\033[0;1;31m'
COLOR_RED_REVERSED_VIDEO='\033[0;7;31m'
COLOR_YELLOW='\033[0;33m'
COLOR_YELLOW_BOLD='\033[0;1;33m'
COLOR_OFF='\033[0m'

FORMAT_BOLD='\033[0;1m'
FORMAT_UNDERLINED='\033[0;4m'
FORMAT_BLINKING='\033[0;5m'
FORMAT_REVERSE_VIDEO='\033[0;7m'

#
# Emit a log message and exit with non-zero return.
#
function die() {
    log_err "$*"
    exit 1
}

#
# Emit a message to stderr.
#
function log_debug() {
    echo -ne "${COLOR_GREY}" 1>&2
    echo -n "$*" 1>&2
    echo -e "${COLOR_OFF}" 1>&2
}

#
# Emit a message to stderr.
#
function log_err() {
    echo -ne "${COLOR_RED_REVERSED_VIDEO}" 1>&2
    echo -n "$*" 1>&2
    echo -e "${COLOR_OFF}" 1>&2
}

#
# Emit a message to stderr.
#
function log_info() {
    echo -ne "${FORMAT_BOLD}" 1>&2
    echo -n "$*" 1>&2
    echo -e "${COLOR_OFF}" 1>&2
}

#
# Emit a message to stderr.
#
function log_warn() {
    echo -ne "${COLOR_YELLOW_BOLD}" 1>&2
    echo -n "$*" 1>&2
    echo -e "${COLOR_OFF}" 1>&2
}

#
# Enable trace logging via set -x
# Args
#  1: [Default: $ORT_BUILD_XTRACE] If set to non-empty, enable tracing
#
# shellcheck disable=SC2120
function set_xtrace() {
    local enable="${1:-${ORT_BUILD_XTRACE:-}}"
    if [ -n "${enable}" ]; then
        set -x
    fi
}

#
# Enable bash strict mode and conditionally set -x.
# @see http://redsymbol.net/articles/unofficial-bash-strict-mode/
# @see set_xtrace
#
function set_strict_mode() {
    set -euo pipefail
    shopt -s inherit_errexit
    set_xtrace
}
