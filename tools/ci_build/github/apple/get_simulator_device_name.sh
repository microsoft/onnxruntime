#!/bin/bash

set -e

# Get a suitable simulator device type name.
# This picks one with name containing "iPhone" with the largest minRuntimeVersion value.
xcrun simctl list devicetypes "iPhone" --json | \
  jq --raw-output '.devicetypes | max_by(.minRuntimeVersion) | .name'
