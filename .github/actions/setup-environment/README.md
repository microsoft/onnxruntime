# Shared build environment setup

Use this local composite action **after checking out this repository** to set up
Python, Node.js, or both. Version selection stays with the caller; action revision
pins live in `action.yml`. Always supply a nonempty version for each requested
tool. Omitting both versions does nothing.

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
- uses: ./.github/actions/setup-environment
  with:
    python-version: '3.12'
    architecture: x64
    node-version: '24.x'
```

For Python-only or Node-only setup, omit the other tool's version. `architecture`
applies only to Python and defaults to the upstream runner architecture.
`registry-url` applies only to Node.js; for example, React Native jobs retain
`https://packagefeedproxy.microsoft.io/npm/`.

The action runs Python setup before Node.js setup when both are requested.
Existing callers keep separate invocations where needed to preserve step order,
conditions, and intervening dependency or SDK setup. Caller step names and
conditions remain outside the composite. No migrated caller uses setup outputs,
so this action does not currently expose outputs.

This is tool setup, **not dependency isolation**: it does not create or activate a
virtual environment, install project dependencies, or change runners, containers,
Android tooling, or SDKs. Upstream PATH updates, registry configuration, caching
defaults, and post-job cleanup remain handled by the same pinned upstream actions.
No migrated caller overrides tool caching options. Add forwarding for new options
only when needed, preserving upstream defaults.

The three setup steps in `macos-ci-setup/action.yml` and
`linux-web-init-and-check/action.yml` intentionally remain direct: they pin
setup-python v6.3.0 and setup-node v5.0.0, unlike the v7.1.0 workflow pins.
Migrating them would require a separately reviewed upstream action upgrade or an
additional legacy-version interface.

Local paths resolve against the checked-out workspace, not a reusable workflow's
source repository. The migrated reusable workflows are called locally by this
repository and already check it out. External consumers must check out a revision
containing this action at the workspace root before using those workflows.

## Validation

With Python and PyYAML available, run the focused metadata and routing tests:

```text
python -m unittest discover -s .github/actions/setup-environment -p test_*.py -v
```

Run `actionlint -shellcheck= -pyflakes=` on changed workflows to validate GitHub
Actions syntax and local-action inputs without requiring shellcheck or pyflakes.
These checks do not download toolchains or execute GitHub-hosted actions; Linux,
macOS, Windows, registry authentication, and post-job behavior still need runner
execution before rollout.
