---
description: "Node.js versions and GPU build/test separation for GitHub Actions workflows and local actions."
applyTo: ".github/workflows/*.yml,.github/workflows/*.yaml,.github/actions/**/action.yml,.github/actions/**/action.yaml"
---

# GitHub Actions

## Node.js Versions

Use Node.js `24.x` for `actions/setup-node` unless a job intentionally tests a different supported Node.js version.
Do not copy obsolete versions from existing jobs when creating or splitting workflows; Node.js 20 is end-of-life.
Check the [Node.js release schedule](https://nodejs.org/en/about/previous-releases) when changing the baseline.

The Node.js version installed by `actions/setup-node` is separate from the runtime used to execute a JavaScript action.
For local JavaScript actions, use the supported `runs.using: node24` runtime; composite actions use `runs.using: composite`.

## GPU Build and Test Jobs

Prefer separate compilation and test-execution jobs when only the tests require a GPU. Compile on an established
CPU runner, then run tests in a dependent GPU job using uploaded build artifacts rather than rebuilding.
Use explicit build-only and test-only phases, preserve test coverage and required-check behavior, and transfer all
runtime libraries, fixtures and test data needed by the tests. Relocate generated absolute paths when runners differ.
See the static-plugin compile/test jobs in [windows_webgpu.yml](../workflows/windows_webgpu.yml) for an example.
