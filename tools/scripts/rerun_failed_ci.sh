#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# rerun_failed_ci.sh - Re-run failed/canceled GitHub Actions CI for a pull request.
#
# For the PR's current head commit, this finds every GitHub Actions workflow run
# whose latest attempt ended in a non-success state (failure, canceled, timed out,
# etc.) and re-runs it -- but ONLY when no newer run for that same workflow is
# already queued or in progress. This avoids piling duplicate runs on a workflow
# that someone (or a previous invocation of this script) has already re-triggered.
#
# Usage:
#   ./rerun_failed_ci.sh <pr-number> [owner/repo] [--dry-run]
#
# Examples:
#   ./rerun_failed_ci.sh 29719
#   ./rerun_failed_ci.sh 29719 microsoft/onnxruntime
#   ./rerun_failed_ci.sh 29719 --dry-run
#
# Requirements: gh (GitHub CLI), authenticated. (Uses gh's built-in --jq; no
# external jq binary required.)
#
# Note: "cancelled" (double l) below is the literal value GitHub's API returns
# for a canceled run's `conclusion`; it must match the API and is not a typo.
# cspell:ignore cancelled
set -euo pipefail

usage() {
  echo "Usage: $0 <pr-number> [owner/repo] [--dry-run]" >&2
  exit 2
}

PR_NUMBER=""
REPO=""
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage ;;
    */*)       REPO="$arg" ;;
    *[!0-9]*)  echo "Unrecognized argument: $arg" >&2; usage ;;
    *)         PR_NUMBER="$arg" ;;
  esac
done

[[ -n "$PR_NUMBER" ]] || usage

command -v gh >/dev/null 2>&1 || { echo "error: gh (GitHub CLI) is not installed" >&2; exit 1; }

# Auto-detect the repository from the current directory when not supplied.
if [[ -z "$REPO" ]]; then
  REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
fi

# Resolve the PR's current head commit; all CI decisions are made against this SHA.
HEAD_SHA="$(gh pr view "$PR_NUMBER" --repo "$REPO" --json headRefOid --jq .headRefOid)"
echo "Repo:    $REPO"
echo "PR:      #$PR_NUMBER"
echo "Head:    $HEAD_SHA"
echo

# Pull every workflow run for this exact commit and decide what to re-run.
# Group runs by workflow. jq's group_by sorts by the key internally, so the
# time-ordered (interleaved) input from `gh run list` does not need to be
# pre-sorted; every run of a given workflow lands in a single group. For each
# workflow:
#   - if any run is still active (queued/in_progress/...), skip it (a rerun already exists);
#   - otherwise take the most recent run and, if it did not succeed, mark it for rerun.
#
# Output: one "<databaseId>\t<workflowName>\t<conclusion>" line per run to rerun.
# gh embeds jq (--jq), so no external jq binary is required.
# shellcheck disable=SC2016  # the single-quoted jq program is intentional; $s/\(...) are jq, not shell
TO_RERUN="$(gh run list --repo "$REPO" --commit "$HEAD_SHA" --limit 300 \
  --json databaseId,workflowName,name,status,conclusion,createdAt \
  --jq '
  def active:      ["queued","in_progress","requested","pending","waiting"];
  def needs_rerun: ["failure","cancelled","timed_out","startup_failure","action_required","stale"];
  group_by(.workflowName)[]
  | { latest:    (sort_by(.createdAt) | last),
      has_active: (any(.[]; .status as $s | active | index($s) != null)) }
  | select(.has_active | not)
  | .latest
  | select(.conclusion as $c | needs_rerun | index($c) != null)
  | "\(.databaseId)\t\(.workflowName)\t\(.conclusion)"
  ')"

if [[ -z "$TO_RERUN" ]]; then
  echo "No failed workflows to re-run (either all passed, or reruns are already queued/in progress)."
  exit 0
fi

echo "Workflows to re-run:"
echo "$TO_RERUN" | while IFS=$'\t' read -r id name conclusion; do
  printf '  %-45s (%s)  run %s\n' "$name" "$conclusion" "$id"
done
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: no reruns triggered."
  exit 0
fi

rerun_count=0
fail_count=0
while IFS=$'\t' read -r id name conclusion; do
  [[ -n "$id" ]] || continue
  # Prefer re-running only the failed jobs; fall back to a full rerun for runs
  # (e.g. fully canceled ones) that have no discrete "failed" jobs to target.
  if gh run rerun "$id" --repo "$REPO" --failed >/dev/null 2>&1 \
     || gh run rerun "$id" --repo "$REPO" >/dev/null 2>&1; then
    echo "  re-ran: $name (run $id)"
    rerun_count=$((rerun_count + 1))
  else
    echo "  FAILED to re-run: $name (run $id)" >&2
    fail_count=$((fail_count + 1))
  fi
done <<<"$TO_RERUN"

echo
echo "Done: $rerun_count re-run, $fail_count failed."
[[ "$fail_count" -eq 0 ]]
