const core = require('@actions/core');
const exec = require('@actions/exec');
const artifact = require('@actions/artifact');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

/**
 * Executes a command and logs its output. Throws an error if the command fails.
 * @param {string} command The command to execute.
 * @param {string[]} [args] Command arguments.
 * @param {exec.ExecOptions} [options] Execution options.
 * @returns {Promise<{exitCode: number, stdout: string, stderr: string}>} Command output.
 */
async function runCommand(command, args = [], options = {}) {
    const effectiveOptions = {
        cwd: process.env.GITHUB_WORKSPACE, // Default working directory
        ignoreReturnCode: false, // Throw error on failure by default
        silent: false, // Show command output by default
        listeners: {
            stdout: (data) => { core.info(data.toString().trim()); },
            stderr: (data) => { core.warning(data.toString().trim()); } // Log stderr as warning by default
        },
        ...options
    };
    const cwdString = effectiveOptions.cwd === process.env.GITHUB_WORKSPACE ? 'default workspace' : effectiveOptions.cwd;
    core.info(`Executing in ${cwdString}: ${command} ${args.map(arg => arg.includes(' ') ? `"${arg}"` : arg).join(' ')}`); // Basic quoting for display
    try {
        const { exitCode, stdout, stderr } = await exec.getExecOutput(command, args, effectiveOptions);

        if (exitCode !== 0 && !effectiveOptions.ignoreReturnCode) {
             core.error(`Stderr: ${stderr}`);
             throw new Error(`Command exited with code ${exitCode}: ${command} ${args.join(' ')}`);
        }
        core.info(`Finished: ${command} ${args.join(' ')}`);
        return { exitCode, stdout, stderr };
    } catch (error) {
        core.error(`Error executing command: ${command} ${args.join(' ')} in ${cwdString}`);
        core.error(error);
        throw new Error(`Command execution failed: ${error.message || error}`);
    }
}

/**
 * Main function for the GitHub Action.
 */
async function main() {
    core.info('Starting Minimal ORT Build Action...');

    // --- Get Inputs ---
    const reducedOpsConfigFileBase = core.getInput('reduced-ops-config-file', { required: true });
    const enableTypeReduction = core.getBooleanInput('enable-type-reduction');
    const enableCustomOps = core.getBooleanInput('enable-custom-ops');
    const skipModelTests = core.getBooleanInput('skip-model-tests');
    const binarySizeReportNamePrefix = core.getInput('binary-size-report-name-prefix');

    // --- Define Paths ---
    const buildDir = process.env.RUNNER_TEMP;
    const workspaceDir = process.env.GITHUB_WORKSPACE;
    if (!buildDir || !workspaceDir) {
        throw new Error("Required environment variables RUNNER_TEMP or GITHUB_WORKSPACE not set.");
    }

    const testDataDownloadDir = path.join(buildDir, '.test_data');
    const fullReducedOpsConfigFile = path.join(testDataDownloadDir, reducedOpsConfigFileBase);
    const debugOutputDir = path.join(buildDir, 'Debug'); // Consistent with build.py default
    const testRunnerPath = path.join(debugOutputDir, 'onnx_test_runner');
    const libraryPath = path.join(debugOutputDir, 'libonnxruntime.so'); // Assuming Linux build
    const binarySizeReportPath = path.join(debugOutputDir, 'binary_size_data.txt');

    core.info(`Using Build Directory (RUNNER_TEMP): ${buildDir}`);
    core.info(`Workspace Directory (GITHUB_WORKSPACE): ${workspaceDir}`);
    core.info(`Test Data Download Directory: ${testDataDownloadDir}`);
    core.info(`Reduced Ops Config File Path: ${fullReducedOpsConfigFile}`);

    const artifactClient = artifact.create();

    try {
        // --- Download Test Data ---
        core.startGroup('Download Test Data Artifact');
        core.info(`Downloading artifact 'test_data' to ${testDataDownloadDir}...`);
        const downloadResponse = await artifactClient.downloadArtifact('test_data', testDataDownloadDir, {
            createArtifactFolder: false // Download directly into the target dir
        });
        core.info(`Artifact download finished. Path: ${downloadResponse.downloadPath}`);
        // Verify the specific config file exists after download
        await fs.access(fullReducedOpsConfigFile, fs.constants.R_OK);
        core.info(`Verified reduced ops config file exists: ${fullReducedOpsConfigFile}`);
        core.endGroup();

        // --- Install Python Requirements ---
        core.startGroup('Install Python Requirements');
        const requirementsPath = path.join(workspaceDir, 'tools/ci_build/github/linux/python/requirements.txt');
        await runCommand('python3', ['-m', 'pip', 'install', '-r', requirementsPath], { cwd: workspaceDir });
        core.endGroup();

        // --- Build Minimal ORT ---
        core.startGroup('Build Minimal ORT');
        const buildScript = path.join(workspaceDir, 'tools/ci_build/build.py');
        const minimalBuildArgs = enableCustomOps ? 'custom_ops' : ''; // build.py handles empty string ok
        const buildArgs = [
            buildScript,
            '--build_dir', buildDir,
            '--cmake_generator', 'Ninja',
            '--config', 'Debug',
            '--skip_submodule_sync',
            '--build_shared_lib',
            '--parallel',
            '--use_binskim_compliant_compile_flags',
            '--minimal_build', minimalBuildArgs, // Add custom_ops if enabled
            '--disable_ml_ops',
            '--include_ops_by_config', fullReducedOpsConfigFile,
        ];
        if (enableTypeReduction) {
            buildArgs.push('--enable_reduced_operator_type_support');
        }
        await runCommand('python3', buildArgs.filter(arg => arg !== ''), { cwd: workspaceDir }); // Filter empty string from minimalBuildArgs
        core.endGroup();

        // --- Run E2E Model Tests ---
        if (!skipModelTests) {
            core.startGroup('Run E2E Model Tests');
            const e2eTestDataDir = path.join(workspaceDir, 'onnxruntime/test/testdata/ort_minimal_e2e_test_data');
            await runCommand(testRunnerPath, [e2eTestDataDir]);
            core.endGroup();
        } else {
            core.info('Skipping E2E model tests as requested.');
        }

        // --- Check Binary Size ---
        core.startGroup('Check Binary Size');
        const checkSizeScript = path.join(workspaceDir, 'tools/ci_build/github/linux/ort_minimal/check_build_binary_size.py');
        const arch = os.machine() === 'x86_64' ? 'x64' : os.machine(); // Simple arch mapping
        const osName = os.platform() === 'linux' ? 'Linux' : os.platform(); // Simple OS mapping
        await runCommand('python3', [
            checkSizeScript,
            '--arch', arch,
            '--os', osName,
            '--build_config', 'minimal-reduced', // As per original script
            libraryPath
        ]);
        core.endGroup();

        // --- Upload Binary Size Report ---
        core.startGroup('Upload Binary Size Report');
        const artifactName = `<span class="math-inline">\{binarySizeReportNamePrefix\}binary\_size\_report\_</span>{arch}_${osName}`;
        core.info(`Uploading ${binarySizeReportPath} as artifact: ${artifactName}`);
        await artifactClient.uploadArtifact(artifactName, [binarySizeReportPath], debugOutputDir, {
             continueOnError: false
        });
        core.endGroup();

        core.info('Action finished successfully.');

    } catch (error) {
        core.setFailed(`Action failed: ${error.message}`);
    }
}