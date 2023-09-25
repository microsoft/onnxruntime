#!/bin/bash

# Note: This script is intended to be called from the iOS packaging build or a similar context
# See tools/ci_build/github/azure-pipelines/mac-ios-packaging-pipeline.yml

set -e
set -x

USAGE_TEXT="Usage: ${0} <binaries staging directory> <artifacts staging directory> <ORT pod name> <ORT pod version>"

abspath() {
  local INPUT_PATH=${1:?"Expected path as the first argument."}
  echo "$(cd "$(dirname "${INPUT_PATH}")" && pwd)/$(basename "${INPUT_PATH}")"
}

# staging directory for binaries (source)
BINARIES_STAGING_DIR=$(abspath "${1:?${USAGE_TEXT}}")
# staging directory for build artifacts (destination)
ARTIFACTS_STAGING_DIR=$(abspath "${2:?${USAGE_TEXT}}")
POD_NAME=${3:?${USAGE_TEXT}}
ORT_POD_VERSION=${4:?${USAGE_TEXT}}

POD_ARCHIVE_BASENAME="pod-archive-${POD_NAME}-${ORT_POD_VERSION}.zip"
PODSPEC_BASENAME="${POD_NAME}.podspec"

pushd "${BINARIES_STAGING_DIR}/${POD_NAME}"

# assemble the files in the artifacts staging directory
zip -r "${ARTIFACTS_STAGING_DIR}/${POD_ARCHIVE_BASENAME}" ./* --exclude "${PODSPEC_BASENAME}"
cp "${PODSPEC_BASENAME}" "${ARTIFACTS_STAGING_DIR}/${PODSPEC_BASENAME}"

popd
