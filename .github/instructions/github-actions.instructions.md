---
description: "GPU build/test separation for GitHub Actions workflows and local actions."
applyTo: ".github/workflows/*.yml,.github/workflows/*.yaml,.github/actions/**/action.yml,.github/actions/**/action.yaml"
---

# GitHub Actions

## GPU Build and Test Jobs

Prefer separate compilation and test-execution jobs when only the tests require a GPU. Compile on an established
CPU runner, then run tests in a dependent GPU job using uploaded build artifacts rather than rebuilding.
Use explicit build-only and test-only phases, preserve test coverage and required-check behavior, and transfer all
runtime libraries, fixtures and test data needed by the tests. Relocate generated absolute paths when runners differ.
See the static-plugin compile/test jobs in [windows_webgpu.yml](../workflows/windows_webgpu.yml) for an example.
