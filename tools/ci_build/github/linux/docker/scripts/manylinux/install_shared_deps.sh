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

echo "Installing azcopy"
mkdir -p /tmp/azcopy
GetFile https://aka.ms/downloadazcopy-v10-linux /tmp/azcopy/azcopy.tar.gz
tar --strip 1 -xf /tmp/azcopy/azcopy.tar.gz -C /tmp/azcopy
cp /tmp/azcopy/azcopy /usr/bin

echo "Installing Node.js"
# The EOL for nodejs v18.17.1 LTS is April 2025
GetFile https://nodejs.org/dist/v18.17.1/node-v18.17.1-linux-x64.tar.gz /tmp/src/node-v18.17.1-linux-x64.tar.gz
tar --strip 1 -xf /tmp/src/node-v18.17.1-linux-x64.tar.gz -C /usr
