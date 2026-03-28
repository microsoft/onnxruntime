const { withAppBuildGradle, withDangerousMod, withXcodeProject, IOSConfig, createRunOncePlugin } = require('@expo/config-plugins');
const generateCode = require('@expo/config-plugins/build/utils/generateCode');
const pkg = require('onnxruntime-react-native/package.json');
const path = require('path');
const fs = require('fs');

const logger = {
  log: (...args) => console.log('[onnxruntime-react-native]', ...args),
  warn: (...args) => console.warn('[onnxruntime-react-native]', ...args),
};

/**
 * Resolve model paths relative to the project root and validate they exist.
 * Returns an array of { src, filename } for files that exist.
 *
 * Models are flattened to their basename (directory structure is discarded).
 * Expo's addResourceFileToGroup adds files as Xcode group members, which are
 * always copied to the .app bundle root. Only folder references preserve
 * subdirectory structure, and the Expo config-plugins API does not create
 * those. We flatten on both platforms for consistent runtime paths.
 *
 * @see https://github.com/expo/expo/blob/main/packages/%40expo/config-plugins/src/ios/utils/Xcodeproj.ts
 * @see https://developer.apple.com/documentation/bundleresources/placing-content-in-a-bundle
 */
function resolveAndValidateModels(projectRoot, modelPaths) {
  const resolved = [];
  const seenFilenames = new Set();

  for (const modelPath of modelPaths) {
    const absolutePath = path.resolve(projectRoot, modelPath);
    const filename = path.basename(absolutePath);

    if (!fs.existsSync(absolutePath)) {
      logger.warn(`Model file not found: ${modelPath} (resolved to ${absolutePath})`);
      continue;
    }

    if (seenFilenames.has(filename)) {
      logger.warn(
        `Multiple models resolve to the same filename "${filename}".`,
        'The last occurrence will overwrite earlier ones because model files are',
        'flattened to their basename for iOS bundle compatibility.',
        'If this is unintentional, rename one of the conflicting models.'
      );
    }
    seenFilenames.add(filename);

    resolved.push({ src: absolutePath, filename });
  }

  return resolved;
}

/**
 * Warn if the Metro config is missing 'onnx' or 'ort' in resolver.assetExts.
 */
function warnIfMetroAssetExtsMissing(projectRoot) {
  const candidates = ['metro.config.js', 'metro.config.cjs'];
  let configPath;
  for (const name of candidates) {
    const p = path.join(projectRoot, name);
    if (fs.existsSync(p)) { configPath = p; break; }
  }
  if (!configPath) return;

  let metroConfig;
  try {
    metroConfig = require(configPath);
  } catch {
    return;
  }

  const assetExts = metroConfig?.resolver?.assetExts;
  if (!Array.isArray(assetExts)) return;

  const missing = ['onnx', 'ort'].filter(ext => !assetExts.includes(ext));
  if (missing.length > 0) {
    logger.warn(
      `Metro assetExts is missing: ${missing.join(', ')}.`,
      'Add them to resolver.assetExts in your metro.config.js so require() can resolve model files in development.',
      'See: https://github.com/microsoft/onnxruntime/blob/main/js/react_native/README.md'
    );
  }
}

/**
 * Copy model files into android/app/src/main/assets/.
 */
function withOrtAndroidModels(config, models) {
  return withDangerousMod(config, [
    'android',
    (config) => {
      const projectRoot = config.modRequest.projectRoot;
      const resolvedModels = resolveAndValidateModels(projectRoot, models);

      if (resolvedModels.length === 0) {
        logger.warn(
          'No valid model files found. Ensure paths in the "models" config are relative to your project root.',
          'See: https://github.com/microsoft/onnxruntime/blob/main/js/react_native/README.md'
        );
        return config;
      }

      const assetsDir = path.join(
        config.modRequest.platformProjectRoot,
        'app',
        'src',
        'main',
        'assets'
      );

      if (!fs.existsSync(assetsDir)) {
        fs.mkdirSync(assetsDir, { recursive: true });
      }

      for (const model of resolvedModels) {
        const destPath = path.join(assetsDir, model.filename);
        const existed = fs.existsSync(destPath);
        fs.copyFileSync(model.src, destPath);
        logger.log(`Android: ${existed ? 'updated' : 'added'} model ${model.filename} in assets/`);
      }

      return config;
    },
  ]);
}

/**
 * Copy model files into the iOS source root and add them to the Xcode project's
 * "Copy Bundle Resources" build phase.
 */
function withOrtIosModels(config, models) {
  return withXcodeProject(config, (config) => {
    const project = config.modResults;
    const projectRoot = config.modRequest.projectRoot;
    const projectName = config.modRequest.projectName;
    const resolvedModels = resolveAndValidateModels(projectRoot, models);

    if (resolvedModels.length === 0) {
      logger.warn(
        'No valid model files found. Ensure paths in the "models" config are relative to your project root.',
        'See: https://github.com/microsoft/onnxruntime/blob/main/js/react_native/README.md'
      );
      return config;
    }

    const sourceRoot = IOSConfig.Paths.getSourceRoot(projectRoot);

    for (const model of resolvedModels) {
      // Copy model file into ios/<ProjectName>/
      const destPath = path.resolve(sourceRoot, model.filename);
      const existed = fs.existsSync(destPath);
      fs.copyFileSync(model.src, destPath);
      logger.log(`iOS: ${existed ? 'updated' : 'added'} model ${model.filename} in ${projectName}/`);

      // Add to Xcode project's "Copy Bundle Resources" build phase
      const pbxFilePath = `${projectName}/${model.filename}`;
      if (!project.hasFile(pbxFilePath)) {
        config.modResults = IOSConfig.XcodeUtils.addResourceFileToGroup({
          filepath: pbxFilePath,
          groupName: projectName,
          isBuildFile: true,
          project,
        });
        logger.log(`iOS: added ${model.filename} to Xcode project resources`);
      }
    }

    return config;
  });
}

const withOrt = (config, props) => {
  // Add build dependency to gradle file
  config = withAppBuildGradle(config, (config) => {
    if (config.modResults.language === 'groovy') {
      config.modResults.contents = generateCode.mergeContents({
        src: config.modResults.contents,
        newSrc: "    implementation project(':onnxruntime-react-native')",
        tag: 'onnxruntime-react-native',
        anchor: /^dependencies[ \t]*\{$/,
        offset: 1,
        comment: '    // onnxruntime-react-native',
      }).contents;
    } else {
      throw new Error('Cannot add ONNX Runtime maven gradle because the build.gradle is not groovy');
    }

    return config;
  });

  // Add build dependency to pod file
  config = withDangerousMod(config, [
    'ios',
    (config) => {
      const podFilePath = path.join(config.modRequest.platformProjectRoot, 'Podfile');
      const contents = fs.readFileSync(podFilePath, { encoding: 'utf-8' });
      const updatedContents = generateCode.mergeContents({
        src: contents,
        newSrc: "  pod 'onnxruntime-react-native', :path => '../node_modules/onnxruntime-react-native'",
        tag: 'onnxruntime-react-native',
        anchor: /^target.+do$/,
        offset: 1,
        comment: '  # onnxruntime-react-native',
      }).contents;
      fs.writeFileSync(podFilePath, updatedContents);
      return config;
    },
  ]);

  // Bundle model assets into native projects
  const models = props?.models || [];
  if (models.length > 0) {
    warnIfMetroAssetExtsMissing(process.cwd());
    config = withOrtAndroidModels(config, models);
    config = withOrtIosModels(config, models);
  }

  return config;
};

exports.default = createRunOncePlugin(withOrt, pkg.name, pkg.version);

// Exported for testing
exports._resolveAndValidateModels = resolveAndValidateModels;
exports._warnIfMetroAssetExtsMissing = warnIfMetroAssetExtsMissing;
