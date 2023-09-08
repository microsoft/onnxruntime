#!/bin/bash
set -e -x

if [ -f /etc/redhat-release ]; then
    dnf update --refresh -y \
    && dnf install -y dotnet-sdk-6.0
elif [ -f /etc/os-release ]; then
  # Get Ubuntu version
  declare repo_version=$(if command -v lsb_release &> /dev/null; then lsb_release -r -s; else grep -oP '(?<=^VERSION_ID=).+' /etc/os-release | tr -d '"'; fi)
  # Download Microsoft signing key and repository
  wget https://packages.microsoft.com/config/ubuntu/$repo_version/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
  # Install Microsoft signing key and repository
  dpkg -i packages-microsoft-prod.deb
  # Clean up
  rm packages-microsoft-prod.deb
  # Update packages
  apt-get update && apt-get install -y dotnet-sdk-6.0
else
  echo "Unsupported OS"
  exit 1
fi
