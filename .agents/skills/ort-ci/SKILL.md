---
name: ort-ci
description: Trigger, re-run, and unblock ONNX Runtime CI checks on a GitHub pull request. Use this skill when a required check is stuck, missing, failed, or needs re-running. Nearly all ORT CI runs as GitHub Actions workflows; only the "Linux Android Emulator QNN CI Pipeline" remains on Azure Pipelines, plus the bot-driven "license/cla" and "Python format" lint checks. Triggers on: rerun CI, retrigger checks, stuck check, missing pipeline, failed CI, license/cla, Python format failure, Doc Gen CI failure, operator docs out of date, /azp run.
---

# ONNX Runtime CI Management

Workflows for triggering, re-running, and unblocking CI checks on an ONNX Runtime PR.
The repository is `microsoft/onnxruntime`. As of 2026-07, nearly all CI runs as **GitHub
Actions** workflows (~80+ checks per PR). Only **one** required check still runs on **Azure
Pipelines** — `Linux Android Emulator QNN CI Pipeline` (host `aiinfra.visualstudio.com`).
There is also the bot-driven `license/cla` status check.

**Failures are not all the same.** Before touching anything, diagnose each failure (see
[Triage: Diagnose Before Re-running](#triage-diagnose-before-re-running)): most failures need a
**code change** and re-running them just fails again; only genuinely transient (network/disk)
failures should be re-run via §1. §4 (Azure Pipelines) applies only to the single QNN pipeline.

Before doing anything, inspect current state so you do not queue duplicate runs.

### Classify a check's provider

GitHub Actions checks have a non-empty `workflowName` and a `detailsUrl` on `github.com`; the
Azure Pipelines check has an empty `workflowName` and a `detailsUrl` on `aiinfra.visualstudio.com`:

```bash
gh pr view <number> --repo microsoft/onnxruntime --json statusCheckRollup \
  --jq '[.statusCheckRollup[] | {name, host:(.detailsUrl|split("/")[2])}]
        | group_by(.host) | map({host:.[0].host, count:length})'
```

## Gather Context

```bash
# PR metadata + all checks grouped by state
gh pr view <number> --repo microsoft/onnxruntime \
  --json number,title,url,state,isDraft,headRefName,headRefOid,baseRefName,statusCheckRollup

# Head / merge SHAs for external CI
gh api repos/microsoft/onnxruntime/pulls/<number> --jq '{head:.head.sha, merge:.merge_commit_sha}'
```

Inspect `statusCheckRollup` and note, for each requested check, whether it is missing, queued,
in progress, failed, canceled, skipped, or already successful. **Do not re-trigger a check
that is already `queued`/`in_progress`/`SUCCESS`** unless the user explicitly asks.

Quickly list just the failed/pending checks:

```bash
gh pr view <number> --repo microsoft/onnxruntime --json statusCheckRollup \
  --jq '.statusCheckRollup[]
        | {name:(.name//.context), status:(.status//.state), conclusion:.conclusion}
        | select(.conclusion!="SUCCESS" and .status!="SUCCESS")'
```

## Triage: Diagnose Before Re-running

**Never blindly re-run failed CI.** Most failures need a code change and will fail again
identically on re-run. Only *transient* failures should be re-run. The process is: **download
the failed job's log, read the actual error, classify it, then fix or re-run case by case.**

### Step 1 — Download the failed log

For a **GitHub Actions** check (the vast majority), get the run and read only the failed steps:

```bash
HEAD_SHA=$(gh api repos/microsoft/onnxruntime/pulls/<number> --jq .head.sha)

# Map failed checks to their workflow run IDs
gh run list --repo microsoft/onnxruntime --commit "$HEAD_SHA" --limit 100 \
  --json databaseId,workflowName,status,conclusion,url \
  --jq '.[] | select(.conclusion=="failure" or .conclusion=="cancelled")'  # cspell:ignore cancelled -- literal GitHub API value

# Dump just the failed steps of a run (grep for the real error)
gh run view <run_id> --repo microsoft/onnxruntime --log-failed > /tmp/ci_<run_id>.log
grep -nE "error:|FAILED|warning:|Traceback|fatal error|No space left|Could not resolve|timed out" \
  /tmp/ci_<run_id>.log | head -50
```

For the **Azure Pipelines** QNN check, open its `detailsUrl` (a `dev.azure.com` /
`aiinfra.visualstudio.com` build page) and download the job log, or use the Azure DevOps
`.../builds/<buildId>/timeline` + log APIs (see the `ci-failure-retrieval` skill for the exact
requests).

### Step 2 — Classify the failure and act

| # | Failure class | How to recognize it in the log | Action — **re-run or fix?** |
|---|---|---|---|
| 1 | **C/C++ warning-as-error** | `error:` on a `-Werror`/`/WX` line — e.g. implicit type-cast/narrowing (`-Werror=conversion`), `unused variable`/`unused parameter` (`-Werror=unused-*`), sign-compare, maybe-uninitialized | **Fix code.** Re-run will not help. Remove/`[[maybe_unused]]` the unused symbol, add an explicit `static_cast<T>()` / `gsl::narrow_cast<T>()` for the cast, or fix the real logic. Rebuild locally to confirm the warning is gone. |
| 2 | **Test failure** | `[  FAILED  ] Suite.Case` (gtest) or `FAILED test_*.py::... - AssertionError` (pytest); often only on some EPs | **Fix code/test.** If a newly added op test fails only on EPs that don't support the op, restrict the test to supported EPs (e.g. gtest `OpTester::Run(..., {kCpuExecutionProvider, kCudaExecutionProvider})` / `excluded_provider_types`, or skip via `SetUp`), or fix the kernel. Don't re-run unchanged. See the `ort-test` skill. |
| 3 | **Transient / infra failure** | `Could not resolve host`, `Connection timed out`, `429 Too Many Requests`, `No space left on device`, package/download 5xx, agent lost, submodule clone timeout — with **no** compile/test error | **Re-run** (§1). This is the one class that a plain re-run fixes. If it recurs 2–3×, escalate — it may be a real infra/proxy issue, not noise. |
| 4 | **Lint / Python format** | `Python format` check fails; `lintrunner` reports diffs | **Fix code** with `lintrunner -a`, commit, push (§3). Re-run alone won't fix it. |

Rules of thumb:
- A compile `error:` or a `[ FAILED ]`/`FAILED` line means **fix the code** — re-running reruns
  the same failing commit and fails identically.
- Only re-run when the log shows a network/disk/agent problem and **no** compile or assertion error.
- When unsure, download the log and read it; do not guess from the check name alone.
- After a code fix, push a new commit — CI re-runs automatically on the new head SHA; you do not
  need §1.

## 1. Re-run Failed GitHub Actions (transient failures only)

The repo ships a helper that re-runs **only** the GitHub Actions workflows whose latest run
for the PR's current head commit failed/canceled — and skips any workflow that already has a
newer run queued or in progress. Use it **only after triage** confirms the failures are
transient (network/disk/agent) — see [Triage](#triage-diagnose-before-re-running). It is the
safest way to retry those without piling on duplicates.

Script: [tools/scripts/rerun_failed_ci.sh](../../../tools/scripts/rerun_failed_ci.sh)

```bash
# Dry run first — shows what would be re-run, triggers nothing
./tools/scripts/rerun_failed_ci.sh <number> --dry-run

# Actually re-run the failed/canceled workflows for the PR's head commit
./tools/scripts/rerun_failed_ci.sh <number>

# Explicit repo (auto-detected from cwd when omitted)
./tools/scripts/rerun_failed_ci.sh <number> microsoft/onnxruntime
```

It prefers `gh run rerun <id> --failed` (retry only failed jobs) and falls back to a full
rerun for fully canceled runs that have no discrete failed jobs. Requires an authenticated
`gh`. Always run `--dry-run` first and confirm the list looks right before the real run.

To re-run one specific workflow manually:

```bash
gh run list --repo microsoft/onnxruntime --commit <head_sha> --limit 100 \
  --json databaseId,workflowName,status,conclusion,url
gh run rerun <run_id> --repo microsoft/onnxruntime --failed   # only failed jobs
gh run rerun <run_id> --repo microsoft/onnxruntime            # full rerun
```

## 2. Unblock `license/cla` (CLA bot)

The `license/cla` check is posted by Microsoft's CLA bot, **independent of the CI pipelines**.
When it is stuck as *"Expected — Waiting for status to be reported"*, re-trigger only the bot —
no CI jobs are re-run — by posting this comment on the PR:

```bash
gh pr comment <number> --repo microsoft/onnxruntime \
  --body "@microsoft-github-policy-service rerun"
```

Then verify it flips to success:

```bash
gh pr view <number> --repo microsoft/onnxruntime --json statusCheckRollup \
  --jq '.statusCheckRollup[] | select((.name//.context)=="license/cla")
        | {status, conclusion}'
```

Expect `{"status":"COMPLETED","conclusion":"SUCCESS"}`.

## 3. Fix the "Python format" required check

The required **Python format** check (job `lint-python-format` in
[.github/workflows/lint.yml](../../../.github/workflows/lint.yml)) runs
`lintrunner --all-files` and fails on any formatting/lint violation. Re-running it will **not**
help — you must fix the code, commit, and push. See
[docs/Coding_Conventions_and_Standards.md](../../../docs/Coding_Conventions_and_Standards.md#linting)
and the `ort-lint` skill.

```bash
# One-time setup (in an activated Python venv)
pip install -r requirements-lintrunner.txt
lintrunner init

# Auto-fix. Prefer changed files; use --all-files to match CI exactly.
lintrunner -a                    # changed files only
lintrunner -a --all-files        # everything (what CI checks)

# Verify clean (no changes reported == pass)
lintrunner --all-files
```

Then commit and push the formatting fixes; the check re-runs automatically on the new commit:

```bash
git add -u && git commit -m "Fix lint" && git push
```

Notes:
- CI runs `lintrunner --all-files`, so a local `lintrunner -a` on only changed files can miss a
  pre-existing violation the CI reports. If the check still fails, run `--all-files` locally.
- The same job also covers C++ clang-format and other adapters; the fix is the same
  (`lintrunner -a`).

## 4. Trigger the Azure Pipelines check (QNN Android Emulator only)

As of 2026-07, the **only** ORT check still on Azure Pipelines is
`Linux Android Emulator QNN CI Pipeline`. Everything else is GitHub Actions (use §1). Trigger
it through the PR comment integration:

```bash
gh pr comment <number> --repo microsoft/onnxruntime \
  --body "/azp run Linux Android Emulator QNN CI Pipeline"
```

Then wait briefly and check for a reply from `azure-pipelines[bot]`:

```bash
# Note: through the GraphQL `comments` field (what `gh pr view --json comments`
# uses), the bot's author.login is `azure-pipelines` (no `[bot]` suffix), even
# though it surfaces as azure-pipelines[bot] in the UI and the REST API.
gh pr view <number> --repo microsoft/onnxruntime --json comments \
  --jq '.comments[] | select(.author.login=="azure-pipelines")
        | {createdAt, body}' | tail
```

- If the bot replies *"No pipelines are associated with this pull request"*, the pipeline is not
  wired to the comment app — use the direct Azure DevOps API fallback (see the
  `Trigger CI Pipelines` section of the private `gh-pr-management` skill for the
  `dev.azure.com` project/definition discovery and `POST .../runs` payload using
  `refName: refs/pull/<number>/merge` and the PR `merge_commit_sha`).
- Keep it to **one batch `/azp run` comment per attempt**; do not spam repeated comments.

## 5. Fix the "Windows GPU Doc Gen CI" check (operator docs out of date)

The **ONNX Runtime Windows GPU Doc Gen CI** check (workflow
[.github/workflows/windows_gpu_doc_gen.yml](../../../.github/workflows/windows_gpu_doc_gen.yml))
builds ORT and runs `build.py --gen_doc validate`. It **fails when the generated operator docs
no longer match what's committed** — typically after you add/modify an operator or its kernel
registrations but forget to regenerate `docs/ContribOperators.md` / `docs/OperatorKernels.md`.
Re-running will **not** help; you must update the docs.

The easiest fix is to download the regenerated docs the failed job already produced: on failure
the workflow uploads a **single artifact named `updated-docs`** that contains both
`OperatorKernels.md` and `ContribOperators.md` at its top level, so you can replace the committed
copies without building locally.

```bash
HEAD_SHA=$(gh api repos/microsoft/onnxruntime/pulls/<number> --jq .head.sha)

# Find the failed Doc Gen run
run_id=$(gh run list --repo microsoft/onnxruntime --commit "$HEAD_SHA" --limit 100 \
  --json databaseId,workflowName,conclusion \
  --jq '.[] | select(.workflowName|test("Doc Gen")) | select(.conclusion=="failure") | .databaseId' | head -1)

# Confirm the artifact is present (expect: updated-docs)
gh api repos/microsoft/onnxruntime/actions/runs/$run_id/artifacts --jq '.artifacts[].name'

# Download the updated docs straight into docs/ (the artifact holds both .md files)
gh run download "$run_id" --repo microsoft/onnxruntime -n updated-docs --dir docs/
```

Then review, commit, and push — the check re-runs on the new commit:

```bash
git diff --stat docs/ContribOperators.md docs/OperatorKernels.md
git add docs/ContribOperators.md docs/OperatorKernels.md
git commit -m "Update operator docs" && git push
```

Notes:
- The `updated-docs` artifact always contains **both** `OperatorKernels.md` and
  `ContribOperators.md` at its top level; `--dir docs/` drops them into place. Only the file(s)
  that actually changed will show up in `git diff` after extraction.
- Equivalent local alternative: `python tools/ci_build/build.py --config Release --build_dir
  build/Linux --gen_doc` after a build, then commit the regenerated files. Downloading the
  artifact is faster since it avoids a full build.

## Special-Case Check Handling

| Check / job name | Owner | How to unblock |
|---|---|---|
| `license/cla` | CLA bot | Comment `@microsoft-github-policy-service rerun` (§2) |
| `Python format` (`lint-python-format`) | GitHub Actions | Fix with `lintrunner -a`, commit, push (§3) — rerun alone won't fix |
| `ONNX Runtime Windows GPU Doc Gen CI` | GitHub Actions | Download the `updated-docs` artifact into `docs/`, commit, push (§5) — rerun alone won't fix |
| `Optional Lint`, `Optional Lint C++` | GitHub Actions | Non-required reviewdog checks; fix warnings or ignore |
| Most CI (Linux/Windows/Mac/CUDA/TensorRT/WebGPU/Web/Android/iOS, `windows_x64_*`, Builds, PR Checks) | GitHub Actions | `rerun_failed_ci.sh <number>` (§1) |
| `Linux Android Emulator QNN CI Pipeline` | Azure DevOps | `/azp run Linux Android Emulator QNN CI Pipeline` (§4), then API fallback |

## Safety Rules

- **Never** trigger release, publish, official, nightly, signing (`ESRP`), deployment, or
  package-upload pipelines unless the user explicitly asks for that class of pipeline. Treat
  names containing `release`, `publish`, `official`, `nightly`, `sign`, `ESRP`, `production`, or
  `deploy` as high-risk and confirm first.
- Always `--dry-run` the rerun script before the real run.
- Do not re-run green checks or dispatch unrelated workflows.
- Prefer PR merge refs (`refs/pull/<number>/merge`) over branch refs for external CI so the run
  validates the merge result.
- Do not claim success until the provider returns a queued/in-progress/completed run ID or URL,
  or the check flips state in `statusCheckRollup`.

## Verify After Triggering

```bash
gh pr view <number> --repo microsoft/onnxruntime --json statusCheckRollup \
  --jq '.statusCheckRollup | group_by(.status//.state)
        | map({state:(.[0].status//.[0].state), count:length})'
```

Confirm the previously stuck/failed check moved to `queued`/`in_progress` (or `SUCCESS` for the
CLA bot), and that no duplicate runs were created for the same head SHA.
