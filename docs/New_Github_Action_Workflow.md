# CI Pipeline Naming Convention

To improve navigation and consistency within our GitHub Actions workflows, **all new CI pipelines that run on pull requests must have names beginning with `(CI)`**. This naming convention is designed to limit the workflows shown in the GitHub Actions web UI to those from the main branch at the top, reducing clutter from workflows in other branches.

## Rationale

- **Easy Identification:**  
  Prefixing CI pipelines with `(CI)` makes it straightforward to identify and filter Continuous Integration workflows, especially those that run on pull requests.

- **Enhanced Navigation in the Web UI:**  
  By standardizing the naming of CI workflows on the main branch, these pipelines will be shown on top of the pipelines in the GitHub Actions sidebar, reducing the noise from workflows in feature branches.

- **Consistent Naming:**  
  A uniform naming convention helps maintain clarity and consistency across our CI/CD pipelines.

## Guidelines

1. **Naming Format:**  
   All new CI workflows that run on pull requests must start with the prefix `(CI)`.  
   - **Example:** `(CI) Build & Test`, `(CI) Lint & Validate`, `(CI) Integration Tests`

2. **Workflow File Location:**  
   Place your CI pipeline YAML files in the `.github/workflows` directory of the repository.

3. **Review Process:**  
   During code reviews, verify that any new CI pipeline adheres to the naming convention before merging. If a new CI pipeline doesn’t follow the (CI) prefix pattern, it should be fixed by someone from the team. Note that while this naming convention is strongly recommended to ensure that only workflows from the main branch appear at the top of the GitHub Actions sidebar, there is no hard enforcement mechanism in place.

## Implementation Example

Below is an example of a GitHub Actions workflow file following the naming convention:

```yaml
name: "(CI) Build & Test"

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '16'
      - name: Install dependencies
        run: npm install
      - name: Run tests
        run: npm test