#!/bin/bash
# Install azcopy, Ninja, Node.js CCache
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

echo "Installing Ninja"
GetFile https://github.com/ninja-build/ninja/archive/v1.10.0.tar.gz /tmp/src/ninja-linux.tar.gz
tar -zxf ninja-linux.tar.gz
cd ninja-1.10.0
cmake -Bbuild-cmake -H.
cmake --build build-cmake
mv ./build-cmake/ninja /usr/bin

echo "Installing Node.js"
# The EOL for nodejs v18.17.1 LTS is April 2025
GetFile https://nodejs.org/dist/v18.17.1/node-v18.17.1-linux-x64.tar.gz /tmp/src/node-v18.17.1-linux-x64.tar.gz
tar --strip 1 -xf /tmp/src/node-v18.17.1-linux-x64.tar.gz -C /usr

echo "Installing CCache"
mkdir -p /tmp/ccache
GetFile https://github.com/ccache/ccache/releases/download/v4.7.4/ccache-4.7.4-linux-x86_64.tar.xz /tmp/src/ccache-4.7.4-linux-x86_64.tar.xz
tar --strip 1 -xf /tmp/src/ccache-4.7.4-linux-x86_64.tar.xz -C /tmp/ccache
cp /tmp/ccache/ccache /usr/bin
