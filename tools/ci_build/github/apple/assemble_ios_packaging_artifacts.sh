#!/bin/bash

# Note: This script is intended to be called from the iOS packaging build or a similar context
# See tools/ci_build/github/azure-pipelines/mac-ios-packaging-pipeline.yml

set -e
set -x

USAGE_TEXT="Usage: ${0} <binaries staging directory> <artifacts staging directory> <ORT pod version> <whether to upload the package archives, 'true' or 'false'>"

abspath() {
  local INPUT_PATH=${1:?"Expected path as the first argument."}
  echo "$(cd "$(dirname "${INPUT_PATH}")" && pwd)/$(basename "${INPUT_PATH}")"
}

# staging directory for binaries (source)
BINARIES_STAGING_DIR=$(abspath "${1:?${USAGE_TEXT}}")
# staging directory for build artifacts (destination)
ARTIFACTS_STAGING_DIR=$(abspath "${2:?${USAGE_TEXT}}")
ORT_POD_VERSION=${3:?${USAGE_TEXT}}
SHOULD_UPLOAD_ARCHIVES=${4:?${USAGE_TEXT}}

STORAGE_ACCOUNT_NAME="onnxruntimepackages"
STORAGE_ACCOUNT_CONTAINER_NAME='$web'
STORAGE_URL_PREFIX=$(az storage account show --name ${STORAGE_ACCOUNT_NAME} --query "primaryEndpoints.web" --output tsv)

assemble_and_upload_pod() {
  local POD_NAME=${1:?"Expected pod name as first argument."}
  local POD_ARCHIVE_BASENAME="pod-archive-${POD_NAME}-${ORT_POD_VERSION}.zip"
  local PODSPEC_BASENAME="${POD_NAME}.podspec"

  pushd ${BINARIES_STAGING_DIR}/${POD_NAME}

  # assemble the files in the artifacts staging directory
  zip -r ${ARTIFACTS_STAGING_DIR}/${POD_ARCHIVE_BASENAME} * --exclude ${PODSPEC_BASENAME}
  cp ${PODSPEC_BASENAME} ${ARTIFACTS_STAGING_DIR}/${PODSPEC_BASENAME}

  if [[ "${SHOULD_UPLOAD_ARCHIVES}" == "true" ]]; then
    # upload the pod archive and set the podspec source to the pod archive URL
    az storage blob upload \
      --account-name ${STORAGE_ACCOUNT_NAME} --container-name ${STORAGE_ACCOUNT_CONTAINER_NAME} \
      --file ${ARTIFACTS_STAGING_DIR}/${POD_ARCHIVE_BASENAME} --name ${POD_ARCHIVE_BASENAME} \
      --if-none-match "*"

    sed -i "" -e "s|file:///http_source_placeholder|${STORAGE_URL_PREFIX}${POD_ARCHIVE_BASENAME}|" \
      ${ARTIFACTS_STAGING_DIR}/${PODSPEC_BASENAME}
  fi

  popd
}

assemble_and_upload_pod "onnxruntime-mobile-c"
assemble_and_upload_pod "onnxruntime-mobile-objc"
assemble_and_upload_pod "onnxruntime-c"
assemble_and_upload_pod "onnxruntime-objc"

cd ${BINARIES_STAGING_DIR}/objc_api_docs
zip -r ${ARTIFACTS_STAGING_DIR}/objc_api_docs.zip *

cat > ${ARTIFACTS_STAGING_DIR}/readme.txt <<'EOM'
Release TODO:
- publish the podspecs
- publish the Objective-C API documentation
EOM
