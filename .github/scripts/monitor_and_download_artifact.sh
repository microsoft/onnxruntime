#!/usr/bin/env bash
set -euo pipefail

# monitor_and_download_artifact.sh
# Monitors the GitHub Actions workflow runs for the current branch in the user's fork
# If a run succeeds, downloads artifacts to ./artifacts/<run-id>
# If a run fails, attempts to rerun up to RERUN_LIMIT times; after that makes a trivial
# commit to retrigger the workflow and continues until success.

REPO="hongxianzhi/onnxruntime"
BRANCH="ci/add-macos-x64-workflow"
WORKFLOW_NAME="Build macOS x86_64 Release"
POLL_INTERVAL=60
RERUN_LIMIT=3
MAX_HOURS=12

start_time=$(date +%s)
deadline=$((start_time + MAX_HOURS * 3600))

echo "Monitoring workflow '${WORKFLOW_NAME}' on ${REPO} branch ${BRANCH}"

while [ $(date +%s) -lt $deadline ]; do
  # Find latest run for this workflow and branch
  run_info=$(gh run list --repo "$REPO" --workflow "$WORKFLOW_NAME" --branch "$BRANCH" --limit 1 --json databaseId,runNumber,conclusion,status --jq '.[]') || true
  if [ -z "$run_info" ]; then
    echo "No runs found yet. Waiting ${POLL_INTERVAL}s..."
    sleep $POLL_INTERVAL
    continue
  fi

  run_id=$(echo "$run_info" | jq -r '.databaseId')
  run_status=$(echo "$run_info" | jq -r '.status')
  run_conclusion=$(echo "$run_info" | jq -r '.conclusion')
  echo "Found run id=$run_id status=$run_status conclusion=$run_conclusion"

  if [ "$run_status" = "in_progress" ] || [ "$run_status" = "queued" ]; then
    echo "Run in progress. Waiting ${POLL_INTERVAL}s..."
    sleep $POLL_INTERVAL
    continue
  fi

  if [ "$run_conclusion" = "success" ]; then
    echo "Run succeeded. Downloading artifacts..."
    dest_dir="artifacts/run-${run_id}"
    mkdir -p "$dest_dir"
    gh run download "$run_id" --repo "$REPO" --dir "$dest_dir"
    echo "Artifacts saved to $dest_dir"
    exit 0
  fi

  if [ "$run_conclusion" = "failure" ] || [ "$run_conclusion" = "cancelled" ] || [ "$run_conclusion" = "timed_out" ]; then
    echo "Run concluded with: $run_conclusion"
    # Attempt reruns
    reruns=0
    while [ $reruns -lt $RERUN_LIMIT ]; do
      echo "Attempting rerun ($((reruns+1))/$RERUN_LIMIT) for run $run_id"
      gh run rerun "$run_id" --repo "$REPO" && break || true
      reruns=$((reruns+1))
      sleep $POLL_INTERVAL
      # check latest run status again
      run_info=$(gh run list --repo "$REPO" --workflow "$WORKFLOW_NAME" --branch "$BRANCH" --limit 1 --json databaseId,runNumber,conclusion,status --jq '.[]') || true
      run_id=$(echo "$run_info" | jq -r '.databaseId')
      run_status=$(echo "$run_info" | jq -r '.status')
      run_conclusion=$(echo "$run_info" | jq -r '.conclusion')
      if [ "$run_status" = "in_progress" ]; then
        echo "Rerun started. Waiting for it to finish."
        break
      fi
    done

    # Wait until rerun finishes or continue loop to pick it up
    sleep $POLL_INTERVAL

    # Re-evaluate latest run
    run_info=$(gh run list --repo "$REPO" --workflow "$WORKFLOW_NAME" --branch "$BRANCH" --limit 1 --json databaseId,runNumber,conclusion,status --jq '.[]') || true
    run_id=$(echo "$run_info" | jq -r '.databaseId')
    run_status=$(echo "$run_info" | jq -r '.status')
    run_conclusion=$(echo "$run_info" | jq -r '.conclusion')

    if [ "$run_conclusion" = "success" ]; then
      echo "Rerun succeeded. Downloading artifacts..."
      dest_dir="artifacts/run-${run_id}"
      mkdir -p "$dest_dir"
      gh run download "$run_id" --repo "$REPO" --dir "$dest_dir"
      echo "Artifacts saved to $dest_dir"
      exit 0
    fi

    # If reruns exhausted or not successful, make a trivial commit to retrigger
    echo "Reruns didn't succeed. Making trivial commit to retrigger workflow."
    tmpfile=".github/workflows/build-macos-x64.yml"
    git fetch fork || true
    git checkout "$BRANCH"
    # append a comment with timestamp to the workflow file and push
    echo "# retrigger at $(date --iso-8601=seconds)" >> "$tmpfile"
    git add "$tmpfile"
    git commit -m "ci: retrigger macos x64 workflow" || true
    git push fork "$BRANCH"
    echo "Triggered new push to branch $BRANCH. Waiting ${POLL_INTERVAL}s..."
    sleep $POLL_INTERVAL
    continue
  fi

  echo "Run conclusion: $run_conclusion (not handled). Waiting ${POLL_INTERVAL}s..."
  sleep $POLL_INTERVAL
done

echo "Deadline reached without successful run. Exiting with non-zero status."
exit 2
