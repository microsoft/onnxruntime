#!/bin/bash
set -e

echo "Installing rust from https://sh.rustup.rs non interatively."
curl https://sh.rustup.rs -sSf | sh -s -- -y
