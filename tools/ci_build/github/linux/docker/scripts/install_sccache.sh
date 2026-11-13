#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

version=0.19.0
arch="$(uname -m)"
case "$arch" in
    x86_64) checksum=aa73c7270c997894038c0d8f0d297865f57569e1687883322bc431713d4bb9dd ;;
    aarch64) checksum=3e47fc4a71b1ddd1b4294a0de9d34498922fdf41e3cd0bda5df3954e3d6deb9d ;;
    *) echo "Unsupported sccache architecture: $arch" >&2; exit 1 ;;
esac

package="sccache-v${version}-${arch}-unknown-linux-musl"
temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

curl --fail --location --silent --show-error --retry 5 \
    "https://github.com/mozilla/sccache/releases/download/v${version}/${package}.tar.gz" \
    --output "$temp_dir/sccache.tar.gz"
echo "$checksum  $temp_dir/sccache.tar.gz" | sha256sum --check -
tar -xzf "$temp_dir/sccache.tar.gz" -C "$temp_dir" "$package/sccache"
install_dir="${1:-/usr/local/bin}"
install -D -m 0755 "$temp_dir/$package/sccache" "$install_dir/sccache"
actual_version="$("$install_dir/sccache" --version)"
if [[ "$actual_version" != "sccache $version" ]]; then
    echo "Unexpected sccache version: $actual_version" >&2
    exit 1
fi
echo "Installed $actual_version at $install_dir/sccache"
