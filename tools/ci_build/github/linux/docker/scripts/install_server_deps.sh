#!/bin/bash
set -e

echo "Installing Go"
if [ $SYS_LONG_BIT = "64" ]; then
  mkdir -p /tmp/go
  cd /tmp/go
  wget https://dl.google.com/go/go1.12.6.linux-amd64.tar.gz
  tar -C /usr/local -zxf /tmp/go/go1.12.6.linux-amd64.tar.gz
fi


