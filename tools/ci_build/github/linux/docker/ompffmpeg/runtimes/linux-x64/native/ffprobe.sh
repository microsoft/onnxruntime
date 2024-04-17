#!/bin/sh

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# Set the LD_LIBRARY_PATH to include the current directory
export LD_LIBRARY_PATH="$SCRIPT_DIR:$LD_LIBRARY_PATH"

# Run the binary
"$SCRIPT_DIR/ffprobe" "$@"