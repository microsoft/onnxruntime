#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Source in each ephemeral build container, after the caller's toolchain PATH selection.
source build/ort-sccache.env || return
if [[ "$SCCACHE_ENABLED" == "true" ]]; then
  rm -f build/sccache-flushed build/sccache-stats.txt || return
  test -r "$SCCACHE_DIR" || return
  if [[ "$SCCACHE_LOCAL_RW_MODE" == "READ_WRITE" ]]; then
    test -w "$SCCACHE_DIR" || return
  fi

  ort_sccache_finish() {
    local build_status=$?
    trap - EXIT
    set +e
    sccache --show-stats > build/sccache-stats.txt
    local stats_status=$?
    cat build/sccache-stats.txt
    sccache --stop-server
    local stop_status=$?
    if [[ "$stats_status" == 0 && "$stop_status" == 0 ]]; then
      touch build/sccache-flushed
    elif [[ "$build_status" == 0 ]]; then
      build_status=1
    fi
    exit "$build_status"
  }
  trap ort_sccache_finish EXIT
  command -v sccache || return
  sccache --version || return
  sccache --start-server || return
  sccache --zero-stats || return
fi
