#!/bin/bash

set -e

USAGE_TEXT="$0 <name | identifier>"

INFO_TYPE=${1:?${USAGE_TEXT}}

case "${INFO_TYPE}" in
  "name" | "identifier") DEVICE_TYPE_FIELD_NAME="${INFO_TYPE}";;
  *) printf "Unknown type: %s\n%s\n" "${INFO_TYPE}" "${USAGE_TEXT}" > /dev/stderr; exit 1;;
esac

# Get info about a suitable simulator device type.
# This picks a device type with name containing "iPhone" with the largest minRuntimeVersion value.
xcrun simctl list devicetypes "iPhone" --json | \
  jq --raw-output ".devicetypes | max_by(.minRuntimeVersion) | .${DEVICE_TYPE_FIELD_NAME}"
