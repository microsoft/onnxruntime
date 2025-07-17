// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This script is used to download WebAssembly build artifacts from CI pipeline.
//
// The goal of this script is to save time for ORT Web developers. For most TypeScript tasks, there is no change in the
// WebAssembly side, so there is no need to rebuild WebAssembly.
//
// It performs the following operations:
// 1. use GitHub Actions REST API as much as possible to get metadata about the status of build and workflow runs.
//    - query the main branch if no "run" parameter is specified
//    - if "run" is specified, it can be a run ID, PR number, branch name, or commit SHA. Try each possibility.
//
// 2. When the artifact is found, use GitHub CLI (gh) to download the artifacts directly to the dist folder.
//

import fs from 'fs';
import { bootstrap as globalAgentBootstrap } from 'global-agent';
import https from 'https';
import path from 'path';
import minimist from 'minimist';
import { execSync } from 'child_process';

const HELP_MESSAGE = `
pull-prebuilt-wasm-artifacts

Usage:
  npm run pull:wasm [run] [options]

  node ./pull-prebuilt-wasm-artifacts [run] [options]

  Run can be specified in one of the following ways:
    action_run_id
    PR number
    branch name (default: "main")

Options:
  -d  --debug       specify the debug build type of the artifacts to download.
  -l  --latest      if set, will always use the latest build, even if it is not completed yet.
      --webgpu-ep   if set, will use the webgpu EP wasm build instead of the default(JSEP) one.
  -h  --help        print this message and exit
`;

const args = minimist(process.argv.slice(2), {
  alias: {
    debug: ['d'],
    help: ['h'],
    latest: ['l'],
  },
});

if (args.help || args.h) {
  console.log(HELP_MESSAGE);
  process.exit();
}

// Check if GitHub CLI (gh) is installed and available in PATH
try {
  execSync('gh --version', { stdio: 'pipe' }).toString().trim();
} catch (e) {
  console.error('Error: GitHub CLI (gh) is not installed or not in PATH.');
  console.error('Please install it from https://cli.github.com/ and try again.');
  process.exit(1);
}

// in NPM script, the args are parsed as:
// npm [npm_command] [npm_flags] -- [user_flags]
//
// The npm_flags will be parsed and removed by NPM, so they will not be available in process.argv directly. Instead,
// they are available in process.env.npm_config_* variables.
//
// For example, if the user runs the command like this:
// > npm run pull:wasm -- --debug
// In this case, `--debug` will be available in `args.debug`
//
// If the user runs the command like this:
// > npm run pull:wasm --debug
// In this case, `--debug` will be available in `process.env.npm_config_debug`, but not in `args.debug` directly.
//
// The following code checks both the command line arguments and the npm_config_* environment variables to get the correct values.
const debug = args.debug || process.env.npm_config_d || process.env.npm_config_debug;
const latest = args.latest || process.env.npm_config_l || process.env.npm_config_latest;
const webgpuEp = args['webgpu-ep'] || process.env.npm_config_webgpu_ep;

const folderName = (debug ? 'Debug_wasm' : 'Release_wasm') + (webgpuEp ? '_webgpu' : '');
const allowImcomplete = latest;

const run = args._[0]; // The first non-option argument

const GITHUB_ACTION_REQUEST_OPTIONS = {
  headers: {
    'user-agent': 'onnxruntime-web artifact pull',
    accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
  },
};

async function downloadJson(url: string): Promise<any> {
  return new Promise((resolve, reject) => {
    https
      .get(url, GITHUB_ACTION_REQUEST_OPTIONS, (res) => {
        const { statusCode } = res;
        const contentType = res.headers['content-type'];

        if (!statusCode) {
          reject(new Error('No response statud code from server.'));
          return;
        }
        if (statusCode >= 400 && statusCode < 500) {
          resolve(null);
          return;
        } else if (statusCode !== 200) {
          reject(new Error(`Failed to download build list. HTTP status code = ${statusCode}`));
          return;
        }
        if (!contentType || !/^application\/json/.test(contentType)) {
          reject(new Error(`unexpected content type: ${contentType}`));
          return;
        }
        res.setEncoding('utf8');
        let rawData = '';
        res.on('data', (chunk) => {
          rawData += chunk;
        });
        res.on('end', () => {
          try {
            resolve(JSON.parse(rawData));
          } catch (e) {
            reject(e);
          }
        });
        res.on('error', (err) => {
          reject(err);
        });
      })
      .on('error', (err) => {
        reject(err);
      });
  });
}

async function downloadArtifactsForRun(run: any): Promise<void> {
  const data = await downloadJson(run.artifacts_url);

  for (const v of data.artifacts) {
    if (v.name === folderName && !v.expired) {
      console.log(`=== Ready to download artifacts "${folderName}" from run: ${run.id} ===`);

      const WASM_FOLDER = path.join(__dirname, '../dist');
      if (!fs.existsSync(WASM_FOLDER)) {
        fs.mkdirSync(WASM_FOLDER);
      } else {
        // TODO: revise artifacts download
        const filesToDelete = ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'];
        if (!folderName.endsWith('_webgpu')) {
          filesToDelete.push('ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm');
        }
        fs.readdirSync(WASM_FOLDER).forEach((file) => {
          if (filesToDelete.includes(file)) {
            const filePath = path.join(WASM_FOLDER, file);
            console.log(`Deleting old file: ${filePath}`);
            fs.unlinkSync(filePath);
          }
        });
      }

      execSync(`gh run download ${run.id} -n ${folderName} -D "${WASM_FOLDER}" -R Microsoft/onnxruntime`);

      return;
    }
  }

  throw new Error(`No artifact "${folderName}" found for the specified build.`);
}

async function main() {
  // Bootstrap global-agent to honor the proxy settings in
  // environment variables, e.g. GLOBAL_AGENT_HTTPS_PROXY.
  // See https://github.com/gajus/global-agent/blob/v3.0.0/README.md#environment-variables for details.
  globalAgentBootstrap();

  console.log(
    `=== Start to pull WebAssembly artifacts "${folderName}" from CI for ${run ? `Run: "${run}"` : 'main branch'} ===`,
  );

  let sha: string | undefined;

  // If param `run` is specified, we try to figure out what it is.
  if (!run) {
    // API reference: https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28#list-workflow-runs-for-a-workflow
    const mainRunData = await downloadJson(
      `https://api.github.com/repos/Microsoft/onnxruntime/actions/workflows/152051496/runs?branch=main${allowImcomplete ? '' : '&status=success'}&per_page=1&exclude_pull_requests=1`,
    );
    if (mainRunData.workflow_runs.length === 0) {
      throw new Error('No build found');
    }
    const run = mainRunData.workflow_runs[0];
    await downloadArtifactsForRun(run);
  } else {
    // If `run` is a number, it is a run ID or PR number
    const runId = parseInt(run, 10);
    // check if runId only contains digits
    const isRunIdDigitsOnly = /^\d+$/.test(run);
    if (isRunIdDigitsOnly && !isNaN(runId)) {
      // Try to treat it as a run ID
      console.log('  # Trying to treat it as a run ID: ' + runId);
      const runData = await downloadJson(`https://api.github.com/repos/Microsoft/onnxruntime/actions/runs/${runId}`);
      if (runData) {
        console.log(`=== Found run: ${runId} ===`);
        await downloadArtifactsForRun(runData);
        return;
      }

      // If not found, try to treat it as a PR number
      console.log(`  # Run ID ${runId} not found or not accessible.`);

      // API reference: https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#get-a-pull-request
      const prData = await downloadJson(`https://api.github.com/repos/Microsoft/onnxruntime/pulls/${runId}`);
      sha = prData?.head?.sha;
    }

    if (sha) {
      console.log(`  # Found PR #${run} with SHA: ${sha}`);
    } else {
      // Try to treat the run parameter as a branch name or commit SHA
      console.log(`  # Trying to treat "${run}" as a branch name`);

      // First, try as a branch name
      const branchData = await downloadJson(
        `https://api.github.com/repos/Microsoft/onnxruntime/branches/${encodeURIComponent(run)}`,
      );
      if (branchData) {
        sha = branchData.commit.sha;
        console.log(`  # Found branch "${run}" with SHA: ${sha}`);
      } else {
        const isPossibleSha = /^[0-9a-f]{7,40}$/.test(`${run}`.trim());
        if (isPossibleSha) {
          // If not a branch, try as a commit SHA (works with both full and short SHA)
          console.log(`  # Trying to treat "${run}" as a commit SHA`);
          const commitData = await downloadJson(
            `https://api.github.com/repos/Microsoft/onnxruntime/commits/${encodeURIComponent(run)}`,
          );
          if (commitData) {
            sha = commitData.sha;
            console.log(`  # Found commit with full SHA: ${sha}`);
          }
        }
      }

      if (!sha) {
        throw new Error(`Could not identify "${run}" as a run ID, PR number, branch name, or commit SHA`);
      }
    }

    // Now that we have the SHA, query for workflow runs associated with this SHA
    const workflowRunsData = await downloadJson(
      `https://api.github.com/repos/Microsoft/onnxruntime/actions/workflows/152051496/runs?head_sha=${sha}`,
    );

    if (!workflowRunsData || workflowRunsData.workflow_runs.length === 0) {
      throw new Error(`No Web CI workflow runs found for SHA: ${sha}`);
    }

    // Get the latest run
    const latestRun = workflowRunsData.workflow_runs[0];
    console.log(`=== Found run for SHA ${sha}: ${latestRun.html_url} ===`);

    // Download artifacts from this run
    await downloadArtifactsForRun(latestRun);
  }
}

void main();
