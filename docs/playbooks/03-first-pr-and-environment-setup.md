# Playbook 03: First PR and Environment Setup

## Outcome

By the end of this playbook, you will be able to set up a local development environment, make a small change, run targeted validation, and open a reviewable pull request.

This playbook assumes you have already worked through [Playbook 01](01-repo-map-and-architecture-overview.md) and [Playbook 02](02-build-test-and-debug-locally.md).

## Start Here

- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [PR_Guidelines.md](../PR_Guidelines.md)
- [Coding_Conventions_and_Standards.md](../Coding_Conventions_and_Standards.md)

## Prerequisites

- GitHub account and a fork of the ONNX Runtime repo
- C++ toolchain for your platform
- Python 3 for build scripts and tests

## Step 1: Clone and prepare your branch

```bash
git clone https://github.com/<your-user>/onnxruntime.git
cd onnxruntime
git remote add upstream https://github.com/microsoft/onnxruntime.git
git fetch upstream
git checkout -b <short-feature-branch-name>
```

## Step 2: Create and activate a Python virtual environment

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Step 3: Pick a small, low-risk change

Use one of these patterns for a first PR:

- a focused test improvement under [onnxruntime/test](../../onnxruntime/test)
- a small documentation fix under [docs](../)
- a narrow bug fix in one file with one corresponding test

Keep your first PR tight. The guideline in [PR_Guidelines.md](../PR_Guidelines.md) recommends small PRs that are easy to review.

## Step 4: Build with a minimal local loop

First run in a new build directory:

Linux or macOS:

```bash
./build.sh --config RelWithDebInfo --update --build --parallel --skip_tests
```

Windows:

```powershell
.\build.bat --config RelWithDebInfo --update --build --parallel --skip_tests
```

If you only changed existing source files after this point, you can usually skip update:

Linux or macOS:

```bash
./build.sh --config RelWithDebInfo --build --parallel --skip_tests
```

Windows:

```powershell
.\build.bat --config RelWithDebInfo --build --parallel --skip_tests
```

## Step 5: Run targeted tests first

Run only tests related to your change before any broad test pass.

Linux example:

```bash
cd build/Linux/RelWithDebInfo
./onnxruntime_test_all --gtest_filter="*YourArea*"
```

Windows example:

```powershell
cd build\Windows\RelWithDebInfo
.\onnxruntime_test_all.exe --gtest_filter="*YourArea*"
```

For model-level tests, see [Model_Test.md](../Model_Test.md).

## Step 6: Prepare and open the PR

Before opening your PR:

- write a clear PR description with motivation
- include test evidence in the PR description
- keep cosmetic changes separate from functional changes
- ensure comments from reviewers are resolved explicitly

## Common Failure Modes

- Build fails after adding files: rerun with update phase enabled.
- Tests not found in build output: verify config and platform path under build.
- PR too broad: split into smaller PRs with one intent each.

## Ready-to-Submit Checklist

- [ ] Scope is small and focused
- [ ] Local build passes on at least one environment
- [ ] Relevant tests pass locally
- [ ] PR description explains motivation and validation
- [ ] No unrelated refactors bundled into the change