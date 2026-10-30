#!/bin/bash
# Install azcopy, Node.js
set -e -x

# Download a file from internet
function GetFile {
  local uri=$1
  local path=$2
  local force=${3:-false}
  local download_retries=${4:-5}
  local retry_wait_time_seconds=${5:-30}

  if [[ -f $path ]]; then
    if [[ $force = false ]]; then
      echo "File '$path' already exists. Skipping download"
      return 0
    else
      rm -rf $path
    fi
  fi

  if [[ -f $uri ]]; then
    echo "'$uri' is a file path, copying file to '$path'"
    cp $uri $path
    return $?
  fi

  echo "Downloading $uri"
  # Use aria2c if available, otherwise use curl
  if command -v aria2c > /dev/null; then
    aria2c -q -d $(dirname $path) -o $(basename $path) "$uri"
  else
    curl "$uri" -sSL --retry $download_retries --retry-delay $retry_wait_time_seconds --create-dirs -o "$path" --fail
  fi

  return $?
}

cd /tmp/src

echo "Installing Node.js"
NODE_VERSION=22.23.4
NODE_ARCHIVE="/tmp/src/node-v${NODE_VERSION}-linux-x64.tar.gz"
GetFile "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" "$NODE_ARCHIVE"
echo "b8492284c76a54baf58d61668dd0e41ee0fc98adabf9503d30f05d08f55b4406  $NODE_ARCHIVE" | sha256sum -c --strict -
tar --strip 1 -xf "$NODE_ARCHIVE" -C /usr
npm install --global --prefix /usr npm@11.21.1 --no-audit --no-fund
