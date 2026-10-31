---
name: code-review
description: "Review ONNX Runtime pull requests, branches, commits, patches, and working-tree changes for actionable findings and missing tests."
---

# Reviewing ONNX Runtime Code Changes

Use this workflow for code review.

## 1. Establish the review scope

Determine the review base before analyzing the change. Identify the exact commits or working-tree state being reviewed,
then enumerate all changed files.

If no changed files are found, respond with: "No changes detected. Please specify a commit range, branch, or patch to
review."

Read every `.github/instructions/**/*.instructions.md` file whose `applyTo` scope matches a changed path. Also load each
domain skill whose description matches the changed subsystem or behavior.

If no instructions file or domain skill matches a changed path, proceed using general ONNX Runtime and
language-specific conventions.

## 2. Analyze changed behavior

Review the diff in context, including controlling code, relevant callers, and affected tests. Trace inputs through the
changed behavior to externally visible results or failure modes.

Distinguish defects introduced by the change from pre-existing issues.

Ensure coverage of all the following:

- correctness and security;
- ABI and API compatibility;
- memory safety and concurrency;
- portability across supported platforms and configurations;
- missing tests for behavior changed by the diff.

Also report actionable clarity and maintainability concerns, even when they are not correctness defects.

Do not report formatting issues already enforced by repository tooling.

## 3. Report findings

Report each actionable finding separately, using an inline comment when supported. Each finding must include:

- the file and line where the finding applies;
- concrete evidence or a triggering scenario;
- the resulting impact;
- an actionable correction.

If no actionable findings are found, say so explicitly.
