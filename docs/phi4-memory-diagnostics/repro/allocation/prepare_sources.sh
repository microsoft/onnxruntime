#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash prepare_sources.sh CLEAN_BASELINE_SOURCE CLEAN_PATCHED_SOURCE" >&2
  echo "Both must be separate existing clean external checkouts at the pinned GenAI revision." >&2
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
baseline=$1 patched=$2
for source in "$baseline" "$patched"; do
  [[ "$source" == /* ]] || { echo "Absolute source paths required" >&2; exit 2; }
  test -d "$source"
  test "$(git -C "$source" rev-parse HEAD)" = ed5f4e87147731e5b07810f9f5c90103b3603cdf
  test -z "$(git -C "$source" status --porcelain)"
  test "$(realpath "$source")" = "$(git -C "$source" rev-parse --show-toplevel)"
done
test "$(realpath "$baseline")" != "$(realpath "$patched")"
printf '%s  %s\n' \
  1485beda53090cc53f4c1cfc0a19c3938d2f4f720a881632e7398d25eb7f038b "$here/historical-tests-normalized.patch" \
  34723df6327bb3430ff00a05d8329d3c69660ceaf67a7efd3d2b268314124e38 "$here/production.diff" \
  | sha256sum --check --status
for source in "$baseline" "$patched"; do
  git -C "$source" apply --check "$here/historical-tests-normalized.patch"
done
git -C "$patched" apply --check "$here/production.diff"
for source in "$baseline" "$patched"; do
  git -C "$source" apply "$here/historical-tests-normalized.patch"
done
git -C "$patched" apply "$here/production.diff"
echo "Applied identical historical test integration to both trees; production.diff to patched only."
