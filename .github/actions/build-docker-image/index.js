// index.js (or main.js - see package.json "main" field)

const core = require('@actions/core');
const exec = require('@actions/exec');
const github = require('@actions/github');

async function run() {
  try {
    // Get inputs
    const dockerfile = core.getInput('Dockerfile', { required: true });
    const dockerBuildArgs = core.getInput('DockerBuildArgs');  // Defaults to "" if not provided.
    const repository = core.getInput('Repository', { required: true });

    const context = dockerfile.substring(0, dockerfile.lastIndexOf('/')) || '.'; // Extract directory of Dockerfile. or .
    core.info(`Dockerfile context directory: ${context}`);

    // Determine if we should use the cache.  Avoid cache if it's a PR from a fork.
    const isPullRequest = github.context.eventName === 'pull_request';
    const isFork = isPullRequest && github.context.payload.pull_request.head.repo.fork;
    const useCache = !isFork;

    let azLoginRan = false;

    if (useCache) {
      // Log in to Azure
      try {
        await exec.exec('az', ['login', '--identity']);
        azLoginRan = true;
        await exec.exec('az', ['acr', 'login', '-n', 'onnxruntimebuildcache']);
      } catch (error) {
        core.setFailed(`Azure login or ACR login failed: ${error.message}`);
        return; // Exit if login fails.  Critical error.
      }
    }


    // Build arguments for get_docker_image.py
    let scriptArgs = [
      'tools/ci_build/get_docker_image.py',
      '--dockerfile', dockerfile,
      '--context', context,
      '--docker-build-args', dockerBuildArgs,
      '--repository', repository,
    ];

    if (useCache) {
      scriptArgs.push('--container-registry', 'onnxruntimebuildcache');
      scriptArgs.push('--use_imagecache');
    }


    // Execute the Python script.
    try {
      await exec.exec('python3', scriptArgs);
    } catch (error) {
      core.setFailed(`Docker build failed: ${error.message}`);
      return; // Exit if docker build failed.
    }

    // Logout from Azure ACR (if we logged in)
    if (azLoginRan) {
      try {
        await exec.exec('az', ['acr', 'logout', '-n', 'onnxruntimebuildcache']);
      } catch (error) {
        core.warning(`Azure ACR logout failed: ${error.message}`); // Warning, not critical.
      }
    }

    core.info('Docker build completed successfully.');

  } catch (error) {
    core.setFailed(error.message);
  }
}

run();