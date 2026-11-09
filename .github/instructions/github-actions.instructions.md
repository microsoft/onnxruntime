---
description: "Node.js version guidance for GitHub Actions workflows and local actions."
applyTo: ".github/workflows/*.yml,.github/workflows/*.yaml,.github/actions/**/action.yml,.github/actions/**/action.yaml"
---

# GitHub Actions

## Node.js Versions

Use Node.js `24.x` for `actions/setup-node` unless a job intentionally tests a different supported Node.js version.
Do not copy obsolete versions from existing jobs when creating or splitting workflows; Node.js 20 is end-of-life.
Check the [Node.js release schedule](https://nodejs.org/en/about/previous-releases) when changing the baseline.

The Node.js version installed by `actions/setup-node` is separate from the runtime used to execute a JavaScript action.
For local JavaScript actions, use the supported `runs.using: node24` runtime; composite actions use `runs.using: composite`.
