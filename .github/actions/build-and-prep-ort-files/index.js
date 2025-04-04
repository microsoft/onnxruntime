#!/usr/bin/env node

const core = require('@actions/core');
const exec = require('@actions/exec');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

// Helper function to execute shell commands using bash -c
async function runCommand(commandString, options = {}) {
  const effectiveOptions = {
    cwd: process.env.GITHUB_WORKSPACE, // Default working directory is repo root
    ignoreReturnCode: false, // Fail action on non-zero exit code
    silent: false, // Stream output to GitHub Actions log
    ...options
  };
  core.info(`Executing in ${effectiveOptions.cwd}: ${commandString}`);
  try {
    // Use bash -c for easier execution of complex commands from the original script
    // Ensure proper quoting if paths/args have spaces, although GITHUB_WORKSPACE/RUNNER_TEMP usually don't
    const exitCode = await exec.exec('bash', ['-c', commandString], effectiveOptions);
     if (exitCode !== 0 && !effectiveOptions.ignoreReturnCode) {
         // Should not happen if ignoreReturnCode is false, but defensive check
         throw new Error(`Command exited with code ${exitCode}`);
    }
    core.info(`Successfully executed: ${commandString}`);
  } catch (error) {
    core.error(`Error executing command: ${commandString}`);
    // Re-throw the specific error message from exec
    throw new Error(`Command failed: ${error.message}`);
  }
}

// Helper function to find the first file matching a pattern in a directory
async function findFirstFile(dir, pattern) {
    try {
        core.info(`Searching for pattern ${pattern} in directory ${dir}`);
        const files = await fs.readdir(dir);
        for (const file of files) {
            if (pattern.test(file)) {
                core.info(`Found matching file: ${file}`);
                return path.join(dir, file);
            }
        }
        core.warning(`No file found matching pattern ${pattern} in ${dir}`);
        return null; // No file found
    } catch (error) {
        if (error.code === 'ENOENT') {
            core.warning(`Directory not found when searching for pattern ${pattern}: ${dir}`);
            return null;
        }
         core.error(`Error reading directory ${dir}: ${error}`);
        throw error; // Re-throw other errors
    }
}

// Main script logic
async function main() {
  core.startGroup('Starting ORT Full Build and Test File Preparation');

  // --- Use Environment Variables for Paths ---
  const buildDir = process.env.RUNNER_TEMP; // Hardcoded to runner temp dir
  const workspaceDir = process.env.GITHUB_WORKSPACE;
  if (!buildDir || !workspaceDir) {
      throw new Error("RUNNER_TEMP or GITHUB_WORKSPACE environment variable not set.");
  }

  const testDataDir = path.join(buildDir, '.test_data');
  const customOpsTestDataDir = path.join(testDataDir, 'custom_ops_model');
  const debugOutputDir = path.join(buildDir, 'Debug'); // Assuming Debug config from original script
  const wheelDir = path.join(debugOutputDir, 'dist');

  core.info(`Using Build Directory (RUNNER_TEMP): ${buildDir}`);
  core.info(`Workspace Directory (GITHUB_WORKSPACE): ${workspaceDir}`);
  core.info(`Derived Test Data Directory: ${testDataDir}`);

  try {
    // Ensure build and test data directories exist
    await fs.mkdir(testDataDir, { recursive: true });
    await fs.mkdir(debugOutputDir, { recursive: true });
    await fs.mkdir(wheelDir, { recursive: true });
    core.info(`Ensured directories exist: ${testDataDir}, ${debugOutputDir}, ${wheelDir}`);


    // --- Step 1: Install Python Requirements ---
    core.startGroup('Install Python Requirements');
    const requirementsPath = path.join(workspaceDir, 'tools/ci_build/github/linux/python/requirements.txt');
    await runCommand(`python3 -m pip install --user -r "${requirementsPath}"`);
    core.endGroup();

    // --- Step 2: Validate Operator Registrations ---
    core.startGroup('Validate Operator Registrations');
    const validatorScript = path.join(workspaceDir, 'tools/ci_build/op_registration_validator.py');
    await runCommand(`python3 "${validatorScript}"`);
    core.endGroup();

    // --- Step 3: Run Full ORT Build ---
    core.startGroup('Build Full ONNX Runtime');
    const buildScript = path.join(workspaceDir, 'tools/ci_build/build.py');
    const buildCommandArgs = [
      `--build_dir "${buildDir}"`, '--cmake_generator Ninja',
      '--config Debug',
      '--skip_submodule_sync',
      '--parallel', '--use_vcpkg', '--use_vcpkg_ms_internal_asset_cache', '--use_binskim_compliant_compile_flags',
      '--build_wheel',
      '--skip_tests',
      '--enable_training_ops',
      '--use_nnapi',
      '--use_coreml'
    ];
    // Use quotes around paths in the command string for robustness
    await runCommand(`python3 "${buildScript}" ${buildCommandArgs.join(' ')}`);
    core.endGroup();

    // --- Step 4: Install the ORT Python Wheel ---
    core.startGroup('Install ORT Wheel');
    const wheelPattern = /\.whl$/;
    const wheelFile = await findFirstFile(wheelDir, wheelPattern);
    if (!wheelFile) {
        throw new Error(`No wheel file found in ${wheelDir}`);
    }
    await runCommand(`python3 -m pip install --user "${wheelFile}"`);
    core.endGroup();

    // --- Step 5 & 6: Convert E2E ONNX models to ORT format ---
    core.startGroup('Convert E2E Models to ORT Format');
    const e2eTestDataPath = path.join(workspaceDir, 'onnxruntime/test/testdata/ort_minimal_e2e_test_data');
    const convertScript = path.join(workspaceDir, 'tools/python/convert_onnx_models_to_ort.py');
    // Run with script path
    await runCommand(`python3 "<span class="math-inline">\{convertScript\}" "</span>{e2eTestDataPath}"`);
    // Run with installed package module
    await runCommand(`python3 -m onnxruntime.tools.convert_onnx_models_to_ort "${e2eTestDataPath}"`);
    core.endGroup();

    // --- Step 7 & 8: Create required ops config files & Append ---
    core.startGroup('Generate and Append Required Ops Config');
    const createConfigScript = path.join(workspaceDir, 'tools/python/create_reduced_build_config.py');
    const testDataRoot = path.join(workspaceDir, 'onnxruntime/test/testdata');
    const requiredOpsConfig = path.join(testDataDir, 'required_ops.ort_models.config');
    const requiredOpsTypesConfig = path.join(testDataDir, 'required_ops_and_types.ort_models.config');

    // Config without type reduction
    await runCommand(`python3 "<span class="math-inline">\{createConfigScript\}" \-\-format ORT "</span>{testDataRoot}" "${requiredOpsConfig}"`);
    // Config with type reduction
    await runCommand(`python3 "<span class="math-inline">\{createConfigScript\}" \-\-format ORT \-\-enable\_type\_reduction "</span>{testDataRoot}" "${requiredOpsTypesConfig}"`);

    // Append standalone invoker ops
    const standaloneInvokerConfig = path.join(e2eTestDataPath, 'required_ops.standalone_invoker.config');
    core.info(`Appending ${standaloneInvokerConfig} to config files...`);
    const standaloneOpsContent = await fs.readFile(standaloneInvokerConfig, 'utf8');
    await fs.appendFile(requiredOpsConfig, os.EOL + standaloneOpsContent);
    await fs.appendFile(requiredOpsTypesConfig, os.EOL + standaloneOpsContent);
    core.info(`Successfully appended standalone invoker ops.`);
    core.endGroup();

    // --- Step 9: Test conversion with custom ops ---
    core.startGroup('Test Custom Op Model Conversion');
    await fs.mkdir(customOpsTestDataDir, { recursive: true });

    const customOpSrcDir = path.join(workspaceDir, 'onnxruntime/test/testdata/custom_op_library');
    const customOpLibrary = path.join(debugOutputDir, 'libcustom_op_library.so'); // Assuming Linux .so extension

    // Copy relevant .onnx files
    core.info(`Copying custom op models from ${customOpSrcDir} to ${customOpsTestDataDir}`);
    const customOpOnnxFiles = (await fs.readdir(customOpSrcDir)).filter(f => f.endsWith('.onnx'));
     if (customOpOnnxFiles.length === 0) {
        core.warning(`No .onnx files found in ${customOpSrcDir} to copy.`);
     } else {
        for (const file of customOpOnnxFiles) {
            const src = path.join(customOpSrcDir, file);
            const dest = path.join(customOpsTestDataDir, file);
            await fs.copyFile(src, dest);
            core.info(`Copied ${file}`);
        }
     }

    // Run conversion only if library exists
    try {
        await fs.access(customOpLibrary, fs.constants.R_OK);
        core.info(`Found custom op library: ${customOpLibrary}`);
        await runCommand(`python3 "<span class="math-inline">\{convertScript\}" \-\-custom\_op\_library "</span>{customOpLibrary}" "${customOpsTestDataDir}"`);
    } catch (err) {
        core.warning(`Custom op library ${customOpLibrary} not found or not readable. Skipping custom op conversion test.`);
    }
    core.endGroup();

    // --- Step 10: Clean up custom ops test dir ---
    core.startGroup('Cleanup Custom Op Test Directory');
    core.info(`Cleaning up ${customOpsTestDataDir}`);
    await fs.rm(customOpsTestDataDir, { recursive: true, force: true });
    core.endGroup();

    core.info('Action finished successfully.');
    core.endGroup(); // End top-level group

  } catch (error) {
    core.setFailed(`Action failed: ${error.message}`);
    core.endGroup(); // Ensure group is closed on failure
  }
}

// Execute the main function
main();