// index.js (or main.js - see package.json "main" field)

const core = require('@actions/core');
const exec = require('@actions/exec');
const github = require('@actions/github');
const fs = require('node:fs/promises');
const path = require('node:path');

async function run() {
    try {
        // Get inputs
        const dockerfile = core.getInput('Dockerfile', { required: true });
        const dockerBuildArgs = core.getInput('DockerBuildArgs');  // Defaults to "" if not provided.
        const repository = core.getInput('Repository', { required: true });

        const context = dockerfile.substring(0, dockerfile.lastIndexOf('/')) || '.'; // Extract directory of Dockerfile.
        core.info(`Dockerfile context directory: ${context}`);

        // Determine if we should use the cache.  Avoid cache if it's a PR from a fork.
        const isPullRequest = github.context.eventName === 'pull_request';
        const isFork = isPullRequest && github.context.payload.pull_request.head.repo.fork;
        const useCache = !isFork;
        const containerRegistry = "onnxruntimebuildcache"; // Moved this here as it is a constant
        const useContainerRegistry = useCache; //Simplified since they have the same condition

        let azLoginRan = false;

        if (useContainerRegistry) {
            // Log in to Azure
            try {
                await exec.exec('az', ['login', '--identity']);
                azLoginRan = true;
                await exec.exec('az', ['acr', 'login', '-n', containerRegistry]);
            } catch (error) {
                core.setFailed(`Azure login or ACR login failed: ${error.message}`);
                return; // Exit if login fails.  Critical error.
            }
        }

        const fullImageName = useContainerRegistry
            ? `${containerRegistry}.azurecr.io/${repository}:latest`
            : `${repository}:latest`;

        core.info(`Image: ${fullImageName}`);
        const repo_dir = process.env.GITHUB_WORKSPACE
        const script_dir = path.join(repo_dir, "tools", "ci_build")

        // Copy deps.txt if it doesn't exist in the Dockerfile's context.
        const dstDepsFile = path.join(context, 'scripts', 'deps.txt');
        try {
          await fs.access(dstDepsFile);
          core.info("deps.txt already exists. No need to copy")
        }
        catch {
          core.info(`Copying deps.txt to: ${dstDepsFile}`);
          await fs.mkdir(path.dirname(dstDepsFile), { recursive: true }); // Ensure 'scripts' directory exists
          await fs.copyFile(path.join(repo_dir, 'cmake', 'deps.txt'), dstDepsFile);
        }
        let dockerCommand = ['build'];
        if (useContainerRegistry) {
            dockerCommand = ["buildx", "build", "--load"];
            dockerCommand.push('--cache-from', `type=registry,ref=${fullImageName}`);
            dockerCommand.push('--build-arg', 'BUILDKIT_INLINE_CACHE=1');
        } else {
          dockerCommand.push("--pull")
        }


        if (dockerBuildArgs) {
            // Split the dockerBuildArgs string into an array, handling spaces and quotes correctly.
            const argsArray = dockerBuildArgs.split(/\s(?=(?:[^'"`]*(['"`])[^'"`]*\1)*[^'"`]*$)/).filter(Boolean);
            for (const arg of argsArray) {
              dockerCommand.push(arg)
            }

        }

        dockerCommand.push('--tag', fullImageName);
        dockerCommand.push('--file', dockerfile);
        dockerCommand.push(context);


        // Execute the Docker build command.
        try {
            if (useContainerRegistry) {
                await exec.exec('docker', ["--log-level", "error", ...dockerCommand]);
            } else {
                await exec.exec('docker', dockerCommand);
            }
        } catch (error) {
            core.setFailed(`Docker build failed: ${error.message}`);
            return; // Exit if docker build failed.
        }

        // Tag the image with the repository name.
        await exec.exec('docker', ['tag', fullImageName, repository]);

        if (useContainerRegistry) {
          await exec.exec("docker", ["push", fullImageName])
        }
        // Logout from Azure ACR (if we logged in)
        if (azLoginRan) {
            try {
                await exec.exec('az', ['acr', 'logout', '-n', containerRegistry]);
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