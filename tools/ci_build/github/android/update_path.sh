#!/bin/bash

set -x
echo $1
origin_repo=$2
current_repo=$3
find . -name "*$1" | while IFS= read -r pathname; do
    new_pathname=${pathname//$origin_repo/$current_repo}
    d1=$(dirname $new_pathname)
    if [ "$pathname" != "$new_pathname" ]; then
        mkdir -p "$d1"  && mv $pathname $new_pathname
    fi
done
