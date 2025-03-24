# Build Docker Image Action

This GitHub Action builds a Docker image and optionally pushes it to an Azure Container Registry (ACR). It's designed to be used as part of a larger CI/CD workflow for projects that require building and testing within a specific Docker environment.

## Inputs

| Input               | Description                                                                                                                   | Required | Default |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------- | :------: | ------- |
| `Dockerfile`        | Path to the Dockerfile.                                                                                                        |   Yes    |         |
| `DockerBuildArgs`   | Arguments to pass to the `docker build` command.  Should be a single string with space-separated arguments.                  |   No    |  `""`   |
| `Repository`        | The image repository name.                                                                                                     |   Yes    |         |

## Outputs
None

## Usage

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Build Docker Image
        uses: ./.github/actions/build-docker-image  # Path to your action directory
        with:
          Dockerfile: ${{ github.workspace }}/path/to/Dockerfile
          DockerBuildArgs: "--build-arg ARG1=value1 --build-arg ARG2=value2"
          Repository: my-docker-image

# Caching

This action supports caching using an Azure Container Registry (ACR) named `onnxruntimebuildcache`. Caching is automatically enabled unless the workflow is triggered by a pull request originating from a fork. This helps to speed up builds by reusing previously built layers.

## Caching Behavior:

*Enabled*: If caching is enabled (not a fork PR), the action will:
    1.  Log in to Azure using a managed identity (`az login --identity --output none`). The output of the login command is suppressed for security.
    2.  Log in to the ACR (`az acr login -n onnxruntimebuildcache`).
    3.  Use `docker buildx build --load` with `--cache-from` to utilize the cache from the ACR. Build arguments like `BUILDKIT_INLINE_CACHE=1` are automatically included.
    4.  If the workflow is triggered by a push to the `main` branch, the built image will be pushed to the ACR (`docker push`).
    5.  Log out from docker registry via `docker logout`.

-   **Disabled:** If caching is disabled (a fork PR), the action will:
    1.  Perform a regular `docker build` (without `buildx` or caching arguments).
    2.  Not push the image to ACR.
    3.  Skip Azure login and logout.

## Build Instructions (for developers of this action)

This action is written in JavaScript and requires Node.js (version 20.x) and npm to be installed. To build the action for distribution:

1.  **Install Dependencies:** Run `npm install` in the action's directory to install the necessary dependencies:

    ```bash
    npm install
    ```

    Dependencies include:
    *   `@actions/core`
    *   `@actions/exec`
    *   `@actions/github`
    *   `@vercel/ncc`

2.  **Build:** Run the following command to compile the action:

    ```bash
    ncc build index.js -o dist
    ```

    This command uses the `@vercel/ncc` compiler to bundle the code (`index.js`) and all its dependencies into a single file (`dist/index.js`). This single file is what GitHub Actions will execute, improving startup performance and reducing the risk of missing dependencies. The `-o dist` flag specifies that the output should be placed in the `dist` directory.

3.  **Commit `dist/index.js`:** The `dist/index.js` file must be committed to the repository. GitHub Actions uses this compiled file to run the action. Please note the `dist` directory is added to `.gitignore`.
