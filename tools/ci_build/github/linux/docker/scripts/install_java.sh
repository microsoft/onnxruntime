#!/bin/bash
set -e -x

if [ -f /etc/redhat-release ]; then
    dnf install -y java-11-openjdk-devel \
    && dnf clean dbcache
elif [ -f /etc/os-release ]; then
    apt-get update && apt-get install -y openjdk-11-jdk
else
  echo "Unsupported OS"
  exit 1
fi
