# CI Pipeline Naming Convention

To improve navigation and consistency within our GitHub Actions workflows, **all new CI pipelines must have names that begin with `(CI)`**. This naming convention will ensure that Continuous Integration workflows are easily identifiable in the GitHub sidebar and throughout our repository.

## Rationale

- **Easy Identification:**  
  Prefixing CI pipelines with `(CI)` allows developers to quickly locate and differentiate these workflows from other GitHub Actions (such as deployments, releases, or custom scripts).

- **Consistent Naming:**  
  A standardized naming scheme helps maintain order and consistency across the project’s CI/CD pipelines.

- **Enhanced Navigation:**  
  With clear naming conventions, developers can more efficiently filter and manage workflows using GitHub’s search and filtering tools.

## Guidelines

1. **Naming Format:**  
   All new CI workflows must start with the prefix `(CI)`.  
   - **Example:** `(CI) Build & Test`, `(CI) Lint & Validate`, `(CI) Integration Tests`

2. **Workflow File Location:**  
   Place your CI pipeline YAML files in the `.github/workflows` directory of the repository.

4. **Review Process:**  
   During code reviews, verify that any new CI pipeline adheres to the naming convention before merging.

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