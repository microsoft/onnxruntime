#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

filter="${TESTBRIDGE_TEST_ONLY:-}"
if [[ -n "$filter" && ! "$filter" =~ ^[A-Za-z_][A-Za-z_0-9]*(/[A-Za-z_][A-Za-z_0-9]*)?$ ]]; then
  echo "Invalid XCTest filter '$filter': use an exact Class or Class/testMethod." >&2
  exit 2
fi

work="$(mktemp -d "${TEST_TMPDIR:?}/ort-xctest.XXXXXX")"
trap 'rm -rf "$work"' EXIT
bundle="%(test_bundle_path)s"
if [[ "$bundle" == *.xctest ]]; then
  cp -R "$bundle" "$work/"
else
  unzip -q "$bundle" -d "$work"
fi
bundles=("$work"/*.xctest)
if [[ ${#bundles[@]} != 1 || ! -d "${bundles[0]}" ]]; then
  echo "Expected exactly one XCTest bundle in '$bundle'; rebuild //objectivec:inference_tests." >&2
  exit 2
fi

environment=(env)
test_environment="%(test_env)s"
if [[ -n "$test_environment" ]]; then
  IFS=',' read -r -a assignments <<< "$test_environment"
  environment+=("${assignments[@]}")
fi
status=0
"${environment[@]}" xcrun xctest -XCTest "${filter:-All}" "${bundles[0]}" 2>&1 | tee "$work/output.log" || status=$?
if [[ "$status" != 0 ]]; then
  exit "$status"
fi
if ! grep -Eq 'Executed [1-9][0-9]* tests?,' "$work/output.log"; then
  echo "XCTest executed no tests for filter '${filter:-<all>}'; select an existing Class/testMethod." >&2
  exit 1
fi
