#!/bin/bash
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
mkdir -p /tmp/src

cd /tmp/src
CPU_ARCH=$(uname -m)


echo "Installing Node.js"
CPU_ARCH=`uname -m`
if [[ "$CPU_ARCH" = "x86_64" ]]; then
  NODEJS_ARCH=x64
elif [[ "$CPU_ARCH" = "aarch64" ]]; then
  NODEJS_ARCH=arm64
else
  NODEJS_ARCH=$CPU_ARCH
fi
GetFile https://nodejs.org/dist/v22.17.1/node-v22.17.1-linux-${NODEJS_ARCH}.tar.gz /tmp/src/node-v22.17.1-linux-${NODEJS_ARCH}.tar.gz
tar --strip 1 -xf /tmp/src/node-v22.17.1-linux-${NODEJS_ARCH}.tar.gz -C /usr

cd /
rm -rf /tmp/src
