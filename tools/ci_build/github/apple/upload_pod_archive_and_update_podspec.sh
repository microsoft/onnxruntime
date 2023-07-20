#!/bin/bash

# Note: This script is intended to be called from the iOS CocoaPods package release pipeline or a similar context.

set -e
set -x

USAGE_TEXT="Usage: ${0} <path to pod archive> <path to podspec>"

abspath() {
  local INPUT_PATH=${1:?"Expected path as the first argument."}
  echo "$(cd "$(dirname "${INPUT_PATH}")" && pwd)/$(basename "${INPUT_PATH}")"
}

POD_ARCHIVE_PATH=$(abspath "${1:?${USAGE_TEXT}}")
PODSPEC_PATH=$(abspath "${2:?${USAGE_TEXT}}")

POD_ARCHIVE_BASENAME=$(basename "${POD_ARCHIVE_PATH}")

STORAGE_ACCOUNT_NAME="onnxruntimepackages"
STORAGE_ACCOUNT_CONTAINER_NAME="\$web"
STORAGE_URL_PREFIX=$(az storage account show --name ${STORAGE_ACCOUNT_NAME} --query "primaryEndpoints.web" --output tsv)

# upload the pod archive and set the podspec source to the pod archive URL
az storage blob upload \
  --account-name ${STORAGE_ACCOUNT_NAME} --container-name ${STORAGE_ACCOUNT_CONTAINER_NAME} \
  --file "${POD_ARCHIVE_PATH}" --name "${POD_ARCHIVE_BASENAME}" \
  --if-none-match "*"

sed -i "" -e "s|file:///http_source_placeholder|${STORAGE_URL_PREFIX}${POD_ARCHIVE_BASENAME}|" "${PODSPEC_PATH}"
