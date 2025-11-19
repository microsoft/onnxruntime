// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const fs = require('fs');
const https = require('https');
const { execFileSync } = require('child_process');
const path = require('path');
const os = require('os');
const AdmZip = require('adm-zip'); // Use adm-zip instead of spawn

async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode !== 200) {
          file.close();
          fs.unlinkSync(dest);
          reject(new Error(`Failed to download from ${url}. HTTP status code = ${res.statusCode}`));
          return;
        }

        res.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
        file.on('error', (err) => {
          fs.unlinkSync(dest);
          reject(err);
        });
      })
      .on('error', (err) => {
        fs.unlinkSync(dest);
        reject(err);
      });
  });
}

async function downloadJson(url) {
  return new Promise((resolve, reject) => {
    https
      .get(url, (res) => {
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

async function installPackages(packages, manifests, feeds) {
  // Step.1: resolve packages
  const resolvedPackages = new Map();
  for (const packageCandidates of packages) {
    // iterate all candidates from packagesInfo and try to find the first one that exists
    for (const { feed, version } of packageCandidates.versions) {
      const { type, index } = feeds[feed];
      const pkg = await resolvePackage(type, index, packageCandidates.name, version);
      if (pkg) {
        resolvedPackages.set(packageCandidates, pkg);
        break;
      }
    }
    if (!resolvedPackages.has(packageCandidates)) {
      throw new Error(`Failed to resolve package. No package exists for: ${JSON.stringify(packageCandidates)}`);
    }
  }

  // Step.2: download packages
  for (const [pkgInfo, pkg] of resolvedPackages) {
    const manifestsForPackage = manifests.filter((x) => x.packagesInfo === pkgInfo);
    await pkg.download(manifestsForPackage);
  }
}

async function resolvePackage(type, index, packageName, version) {
  // https://learn.microsoft.com/en-us/nuget/api/overview
  const nugetPackageUrlResolver = async (index, packageName, version) => {
    // STEP.1 - get Nuget package index
    const nugetIndex = await downloadJson(index);
    if (!nugetIndex) {
      throw new Error(`Failed to download Nuget index from ${index}`);
    }

    // STEP.2 - get the base url of "PackageBaseAddress/3.0.0"
    const packageBaseUrl = nugetIndex.resources.find((x) => x['@type'] === 'PackageBaseAddress/3.0.0')?.['@id'];
    if (!packageBaseUrl) {
      throw new Error(`Failed to find PackageBaseAddress in Nuget index`);
    }

    // STEP.3 - get the package version info
    const packageInfo = await downloadJson(`${packageBaseUrl}${packageName.toLowerCase()}/index.json`);
    if (!packageInfo.versions.includes(version.toLowerCase())) {
      throw new Error(`Failed to find specific package versions for ${packageName} in ${index}`);
    }

    // STEP.4 - generate the package URL
    const packageUrl = `${packageBaseUrl}${packageName.toLowerCase()}/${version.toLowerCase()}/${packageName.toLowerCase()}.${version.toLowerCase()}.nupkg`;
    const packageFileName = `${packageName.toLowerCase()}.${version.toLowerCase()}.nupkg`;

    return {
      download: async (manifests) => {
        if (manifests.length === 0) {
          return;
        }

        // Create a temporary directory
        const tempDir = path.join(os.tmpdir(), `onnxruntime-node-pkgs_${Date.now()}`);
        fs.mkdirSync(tempDir, { recursive: true });

        try {
          const packageFilePath = path.join(tempDir, packageFileName);

          // Download the NuGet package
          console.log(`Downloading ${packageUrl}`);
          await downloadFile(packageUrl, packageFilePath);

          // Load the NuGet package (which is a ZIP file)
          let zip;
          try {
            zip = new AdmZip(packageFilePath);
          } catch (err) {
            throw new Error(`Failed to open NuGet package: ${err.message}`);
          }

          // Extract only the needed files from the package
          const extractDir = path.join(tempDir, 'extracted');
          fs.mkdirSync(extractDir, { recursive: true });

          // Process each manifest and extract/copy files to their destinations
          for (const manifest of manifests) {
            const { filepath, pathInPackage } = manifest;

            // Create directory for the target file
            const targetDir = path.dirname(filepath);
            fs.mkdirSync(targetDir, { recursive: true });

            // Check if the file exists directly in the zip
            const zipEntry = zip.getEntry(pathInPackage);
            if (!zipEntry) {
              throw new Error(`Failed to find ${pathInPackage} in NuGet package`);
            }

            console.log(`Extracting ${pathInPackage} to ${filepath}`);

            // Extract just this entry to a temporary location
            const extractedFilePath = path.join(extractDir, path.basename(pathInPackage));
            zip.extractEntryTo(zipEntry, extractDir, false, true);

            // Copy to the final destination
            fs.copyFileSync(extractedFilePath, filepath);
          }
        } finally {
          // Clean up the temporary directory - always runs even if an error occurs
          try {
            fs.rmSync(tempDir, { recursive: true });
          } catch (e) {
            console.warn(`Failed to clean up temporary directory: ${tempDir}`, e);
            // Don't rethrow this error as it would mask the original error
          }
        }
      },
    };
  };

  switch (type) {
    case 'nuget':
      return await nugetPackageUrlResolver(index, packageName, version);
    default:
      throw new Error(`Unsupported package type: ${type}`);
  }
}

function tryGetCudaVersion() {
  // Should only return 11 or 12.

  // try to get the CUDA version from the system ( `nvcc --version` )
  let ver = 12;
  try {
    const nvccVersion = execFileSync('nvcc', ['--version'], { encoding: 'utf8' });
    const match = nvccVersion.match(/release (\d+)/);
    if (match) {
      ver = parseInt(match[1]);
      if (ver !== 11 && ver !== 12) {
        throw new Error(`Unsupported CUDA version: ${ver}`);
      }
    }
  } catch (e) {
    if (e?.code === 'ENOENT') {
      console.warn('`nvcc` not found. Assuming CUDA 12.');
    } else {
      console.warn('Failed to detect CUDA version from `nvcc --version`:', e.message);
    }
  }

  // assume CUDA 12 if failed to detect
  return ver;
}

function parseInstallFlag() {
  let flag = process.env.ONNXRUNTIME_NODE_INSTALL || process.env.npm_config_onnxruntime_node_install;
  if (!flag) {
    for (let i = 0; i < process.argv.length; i++) {
      if (process.argv[i].startsWith('--onnxruntime-node-install=')) {
        flag = process.argv[i].split('=')[1];
        break;
      } else if (process.argv[i] === '--onnxruntime-node-install') {
        flag = 'true';
      }
    }
  }
  switch (flag) {
    case 'true':
    case '1':
    case 'ON':
      return true;
    case 'skip':
      return false;
    case undefined: {
      flag = parseInstallCudaFlag();
      if (flag === 'skip') {
        return false;
      }
      if (flag === 11) {
        throw new Error('CUDA 11 is no longer supported. Please consider using CPU or upgrade to CUDA 12.');
      }
      if (flag === 12) {
        return 'cuda12';
      }
      return undefined;
    }
    default:
      if (!flag || typeof flag !== 'string') {
        throw new Error(`Invalid value for --onnxruntime-node-install: ${flag}`);
      }
  }
}

function parseInstallCudaFlag() {
  let flag = process.env.ONNXRUNTIME_NODE_INSTALL_CUDA || process.env.npm_config_onnxruntime_node_install_cuda;
  if (!flag) {
    for (let i = 0; i < process.argv.length; i++) {
      if (process.argv[i].startsWith('--onnxruntime-node-install-cuda=')) {
        flag = process.argv[i].split('=')[1];
        break;
      } else if (process.argv[i] === '--onnxruntime-node-install-cuda') {
        flag = 'true';
      }
    }
  }
  switch (flag) {
    case 'true':
    case '1':
    case 'ON':
      return tryGetCudaVersion();
    case 'v11':
      return 11;
    case 'v12':
      return 12;
    case 'skip':
    case undefined:
      return flag;
    default:
      throw new Error(`Invalid value for --onnxruntime-node-install-cuda: ${flag}`);
  }
}

module.exports = {
  installPackages,
  parseInstallFlag,
};
