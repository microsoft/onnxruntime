const core = require('@actions/core');
const exec = require('@actions/exec');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

/**
 * Executes a command and logs its output. Throws an error if the command fails.
 * @param {string} command The command to execute.
 * @param {string[]} [args] Command arguments.
 * @param {exec.ExecOptions} [options] Execution options, including 'cwd' for working directory.
 * @returns {Promise<{exitCode: number, stdout: string, stderr: string}>} Command output.
 */
async function runCommand(command, args = [], options = {}) {
    // Default working directory is the workspace root unless overridden
    const defaultCwd = process.env.GITHUB_WORKSPACE;
    const effectiveOptions = {
        cwd: defaultCwd, // Apply default
        ignoreReturnCode: false, // Throw error on failure by default
        silent: false, // Show command output by default
        listeners: {
            stdout: (data) => { core.info(data.toString().trim()); },
            stderr: (data) => { core.warning(data.toString().trim()); } // Log stderr as warning by default
        },
        ...options // Apply overrides from caller (like cwd)
    };

    const cwdString = effectiveOptions.cwd === defaultCwd ? 'default workspace' : effectiveOptions.cwd;
    core.info(`Executing in ${cwdString}: ${command} ${args.map(arg => arg.includes(' ') ? `"${arg}"` : arg).join(' ')}`); // Basic quoting for display

    try {
        const { exitCode, stdout, stderr } = await exec.getExecOutput(command, args, effectiveOptions);

        if (exitCode !== 0 && !effectiveOptions.ignoreReturnCode) {
             // Log stderr specifically as error if command failed
             core.error(`Stderr: ${stderr}`);
             throw new Error(`Command exited with code ${exitCode}: ${command} ${args.join(' ')}`);
        }
        core.info(`Finished: ${command} ${args.join(' ')}`);
        return { exitCode, stdout, stderr };
    } catch (error) {
        // exec.getExecOutput throws on non-zero exit codes if ignoreReturnCode is false
        core.error(`Error executing command: ${command} ${args.join(' ')} in ${cwdString}`);
        core.error(error); // Log the full error object from exec
        // Rethrow with a clearer message if possible
        throw new Error(`Command execution failed: ${error.message || error}`);
    }
}

/**
 * Finds the first file in a directory matching a regex pattern.
 * @param {string} dir Directory path.
 * @param {RegExp} pattern Regex pattern to match.
 * @returns {Promise<string|null>} Full path to the matched file or null.
 */
async function findFirstFile(dir, pattern) {
    try {
        const files = await fs.readdir(dir);
        for (const file of files) {
            if (pattern.test(file)) {
                return path.join(dir, file);
            }
        }
        core.info(`No file matching pattern ${pattern} found in ${dir}.`);
        return null; // No file found
    } catch (error) {
        if (error.code === 'ENOENT') {
            core.warning(`Directory not found when searching for pattern ${pattern}: ${dir}`);
            return null;
        }
        // Log other fs errors before rethrowing
        core.error(`Error reading directory ${dir}: ${error}`);
        throw error;
    }
}

/**
 * Main function for the GitHub Action.
 */
async function main() {
    core.info('Starting ORT Full Build and Test File Preparation Action...');

    // --- Define Paths using Environment Variables ---
    const buildDir = process.env.RUNNER_TEMP; // Hardcoded to runner temp dir
    const workspaceDir = process.env.GITHUB_WORKSPACE;
    if (!buildDir || !workspaceDir) {
        throw new Error("Required environment variables RUNNER_TEMP or GITHUB_WORKSPACE not set.");
    }

    const testDataDir = path.join(buildDir, 'minimal_build_test_data');
    const customOpsTestDataDir = path.join(testDataDir, 'custom_ops_model');
    const debugOutputDir = path.join(buildDir, 'Debug'); // Assuming Debug config based on original script
    const wheelDir = path.join(debugOutputDir, 'dist'); // Directory containing the built wheel

    core.info(`Using Build Directory (RUNNER_TEMP): ${buildDir}`);
    core.info(`Workspace Directory (GITHUB_WORKSPACE): ${workspaceDir}`);
    core.info(`Derived Test Data Directory: ${testDataDir}`);
    core.info(`Wheel Directory: ${wheelDir}`);

    try { // Wrap main logic in try/catch for core.setFailed
        // Ensure necessary directories exist
        await fs.mkdir(testDataDir, { recursive: true });
        await fs.mkdir(debugOutputDir, { recursive: true });
        await fs.mkdir(wheelDir, { recursive: true });
        core.info(`Ensured directories exist: ${testDataDir}, ${debugOutputDir}, ${wheelDir}`);


        // --- Step 1: Install Python Requirements ---
        core.startGroup('Install Python Requirements');
        const requirementsPath = path.join(workspaceDir, 'tools/ci_build/github/linux/python/requirements.txt');
        // Run this in the workspace directory as it reads a file from there
        await runCommand('python3', ['-m', 'pip', 'install', '--user', '-r', requirementsPath], { cwd: workspaceDir });
        core.endGroup();

        // --- Step 2: Validate Operator Registrations ---
        core.startGroup('Validate Operator Registrations');
        const validatorScript = path.join(workspaceDir, 'tools/ci_build/op_registration_validator.py');
        // Run this in the workspace directory as it likely operates on source files
        await runCommand('python3', [validatorScript], { cwd: workspaceDir });
        core.endGroup();

        // --- Step 3: Run Full ORT Build ---
        core.startGroup('Run Full ORT Build');
        const buildScript = path.join(workspaceDir, 'tools/ci_build/build.py');
        const buildArgs = [
            buildScript,
            '--build_dir', buildDir,
            '--cmake_generator', 'Ninja',
            '--config', 'Debug',
            '--skip_submodule_sync',
            '--parallel',
            '--use_vcpkg',
            '--use_vcpkg_ms_internal_asset_cache',
            '--use_binskim_compliant_compile_flags',
            '--build_wheel',
            '--skip_tests',
            '--enable_training_ops',
            '--use_nnapi',
            '--use_coreml'
        ];
        // Run build from the workspace directory
        await runCommand('python3', buildArgs, { cwd: workspaceDir });
        core.endGroup();

        // --- Step 4: Install the ORT Python Wheel ---
        core.startGroup('Install ORT Python Wheel');
        // We can run pip from anywhere, it doesn't need a specific CWD here
        await runCommand('python3', [
            '-m', 'pip', 'install',
            '--user', // Keep installing as user
            '--find-links', wheelDir,
            'onnxruntime' // Specify the package name
        ]);
        core.endGroup();

        // --- Set CWD for subsequent python scripts ---
        // Reason: After installing the 'onnxruntime' wheel, running python scripts
        // from the workspace root might cause Python to import the local 'onnxruntime'
        // source directory instead of the newly installed package. Changing the CWD
        // to a neutral location like '/tmp' prevents this import conflict.
        const pythonScriptCwd = '/tmp';
        await fs.mkdir(pythonScriptCwd, { recursive: true }); // Ensure /tmp exists
        core.info(`Setting CWD for subsequent Python scripts to: ${pythonScriptCwd}`);


        // --- Step 5: Convert E2E ONNX models to ORT format ---
        core.startGroup('Convert E2E ONNX models to ORT format (Tool)');
        const e2eTestDataPath = path.join(workspaceDir, 'onnxruntime/test/testdata/ort_minimal_e2e_test_data');
        const convertScript = path.join(workspaceDir, 'tools/python/convert_onnx_models_to_ort.py');
        // Pass absolute paths and change CWD
        await runCommand('python3', [convertScript, e2eTestDataPath], { cwd: pythonScriptCwd });
        core.endGroup();

        // --- Step 6: Convert again using installed package tool ---
        core.startGroup('Convert E2E ONNX models to ORT format (Installed Package)');
        // Pass absolute paths and change CWD
        await runCommand('python3', ['-m', 'onnxruntime.tools.convert_onnx_models_to_ort', e2eTestDataPath], { cwd: pythonScriptCwd });
        core.endGroup();

        // --- Step 7: Create required ops config files ---
        core.startGroup('Create required ops config files');
        const createConfigScript = path.join(workspaceDir, 'tools/python/create_reduced_build_config.py');
        const testDataRoot = path.join(workspaceDir, 'onnxruntime/test/testdata');
        const requiredOpsConfig = path.join(testDataDir, 'required_ops.ort_models.config');
        const requiredOpsTypesConfig = path.join(testDataDir, 'required_ops_and_types.ort_models.config');

        // Pass absolute paths and change CWD
        // Config without type reduction
        await runCommand('python3', [createConfigScript, '--format', 'ORT', testDataRoot, requiredOpsConfig], { cwd: pythonScriptCwd });

        // Config with type reduction
        await runCommand('python3', [createConfigScript, '--format', 'ORT', '--enable_type_reduction', testDataRoot, requiredOpsTypesConfig], { cwd: pythonScriptCwd });
        core.endGroup();

        // --- Step 8: Append standalone invoker ops ---
        core.startGroup('Append standalone invoker ops');
        const standaloneInvokerConfig = path.join(e2eTestDataPath, 'required_ops.standalone_invoker.config');
        core.info(`Appending ${standaloneInvokerConfig} to config files...`);
        const standaloneOpsContent = await fs.readFile(standaloneInvokerConfig, 'utf8');
        await fs.appendFile(requiredOpsConfig, os.EOL + standaloneOpsContent);
        await fs.appendFile(requiredOpsTypesConfig, os.EOL + standaloneOpsContent);
        core.info(`Successfully appended standalone invoker ops.`);
        core.endGroup();

        // --- Step 9: Test conversion with custom ops ---
        core.startGroup('Test conversion with custom ops');
        await fs.mkdir(customOpsTestDataDir, { recursive: true });

        const customOpSrcDir = path.join(workspaceDir, 'onnxruntime/test/testdata/custom_op_library');
        const customOpLibrary = path.join(debugOutputDir, 'libcustom_op_library.so'); // Assuming Linux .so extension

        // Copy relevant .onnx files
        core.info(`Copying custom op models from ${customOpSrcDir} to ${customOpsTestDataDir}`);
        let filesCopied = 0;
        try {
            const customOpOnnxFiles = (await fs.readdir(customOpSrcDir)).filter(f => f.endsWith('.onnx'));
             if (customOpOnnxFiles.length === 0) {
                core.warning(`No .onnx files found in ${customOpSrcDir} to copy.`);
             } else {
                for (const file of customOpOnnxFiles) {
                    const src = path.join(customOpSrcDir, file);
                    const dest = path.join(customOpsTestDataDir, file);
                    await fs.copyFile(src, dest);
                    core.info(`Copied ${file}`);
                    filesCopied++;
                }
             }
             core.info(`Copied ${filesCopied} custom op model files.`);
        } catch(copyError) {
            core.warning(`Could not copy custom op models from ${customOpSrcDir}: ${copyError.message}`);
        }


        // Run conversion, checking if library exists first
        // Pass absolute paths and change CWD
        try {
            await fs.access(customOpLibrary, fs.constants.R_OK);
            await runCommand('python3', [convertScript, '--custom_op_library', customOpLibrary, customOpsTestDataDir], { cwd: pythonScriptCwd });
        } catch (err) {
            // Log as warning if library doesn't exist or isn't readable
            core.warning(`Custom op library ${customOpLibrary} not found or not readable. Skipping custom op conversion test. Error: ${err.message}`);
        }
        core.endGroup();

        // --- Step 10: Clean up custom ops test dir ---
        core.startGroup('Clean up custom ops test directory');
        core.info(`Cleaning up ${customOpsTestDataDir}`);
        await fs.rm(customOpsTestDataDir, { recursive: true, force: true });
        core.endGroup();

        core.info('Action finished successfully.');

    } catch (error) {
        core.setFailed(`Action failed: ${error.message}`);
    }
}

// Execute the main function
main();