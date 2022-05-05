#!/bin/bash

set -x
echo $1
find . -name "*$1" | while IFS= read -r pathname; do
    new_pathname=${pathname//"/mnt/vss/_work/3/s"/"/Users/runner/work/1/s/"}
    d1=$(dirname $new_pathname)
    if [ "$pathname" != "$new_pathname" ]; then
        mkdir -p "$d1"  && mv $pathname $new_pathname
    fi
done
