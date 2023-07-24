#!/bin/bash

# Note: This script is intended to be called from the iOS CocoaPods package release pipeline or a similar context.

set -x

IFS='' read -d '' -r USAGE_TEXT <<USAGE
Usage: ${0} <pod archive path glob pattern> <podspec path>
  Example pod archive path glob pattern: "./pod-archive-*.zip"
  Quote the pattern to avoid shell expansion.
USAGE

set -e

abspath() {
  local INPUT_PATH=${1:?"Expected path as the first argument."}
  echo "$(cd "$(dirname "${INPUT_PATH}")" && pwd)/$(basename "${INPUT_PATH}")"
}

POD_ARCHIVE_PATH_PATTERN=${1:?${USAGE_TEXT}}
PODSPEC_PATH=$(abspath "${2:?${USAGE_TEXT}}")

# expand pod archive path pattern to exactly one path
POD_ARCHIVE_PATHS=()
while IFS='' read -r line; do POD_ARCHIVE_PATHS+=("$line"); done < <( compgen -G "${POD_ARCHIVE_PATH_PATTERN}" )
if [[ "${#POD_ARCHIVE_PATHS[@]}" -ne "1" ]]; then
  echo "Did not find exactly one pod archive file."
  exit 1
fi

POD_ARCHIVE_PATH=$(abspath "${POD_ARCHIVE_PATHS[0]}")
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
