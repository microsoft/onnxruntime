const configPlugin = require('@expo/config-plugins');
const generateCode = require('@expo/config-plugins/build/utils/generateCode');
const pkg = require('onnxruntime-react-native/package.json');
const path = require('path');
const fs = require('fs');

/**
 * Converts directory path to node_modules to a relative path for the ios/Podfile file.
 * @param {string} iosProjectRoot the ios/ directory path
 * @param {string} packageRoot the node_modules/onnxruntime-react-native directory path
 * @returns {string} the relative path to the packageRoot from the iosProjectRoot
 */
function toPodfileRelativePath(iosProjectRoot, packageRoot) {
  const rel = path.relative(iosProjectRoot, packageRoot);
  return rel ? rel.split(path.sep).join('/') : '.';
}

const withOrt = (config) => {
  // Add build dependency to gradle file
  config = configPlugin.withAppBuildGradle(config, (config) => {
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

  // Register OnnxruntimePackage in MainApplication for New Architecture / Expo prebuild
  config = configPlugin.withMainApplication(config, (config) => {
    const lang = config.modResults.language;
    if (lang === 'kt') {
      config.modResults.contents = generateCode.mergeContents({
        src: config.modResults.contents,
        newSrc: 'import ai.onnxruntime.reactnative.OnnxruntimePackage',
        tag: 'onnxruntime-react-native-import',
        anchor: /^import /m,
        offset: 0,
        comment: '//',
      }).contents;
      config.modResults.contents = generateCode.mergeContents({
        src: config.modResults.contents,
        newSrc: '      add(OnnxruntimePackage())',
        tag: 'onnxruntime-react-native-package',
        anchor: /override fun getPackages\(\)/,
        offset: 2,
        comment: '//',
      }).contents;
    } else if (lang === 'java') {
      config.modResults.contents = generateCode.mergeContents({
        src: config.modResults.contents,
        newSrc: 'import ai.onnxruntime.reactnative.OnnxruntimePackage;',
        tag: 'onnxruntime-react-native-import',
        anchor: /^import /m,
        offset: 0,
        comment: '//',
      }).contents;
      if (!config.modResults.contents.includes('packages.add(new OnnxruntimePackage())')) {
        if (/return\s+new PackageList\(this\)\.getPackages\(\);/.test(config.modResults.contents)) {
          config.modResults.contents = config.modResults.contents.replace(
            /(\s*)return\s+new PackageList\(this\)\.getPackages\(\);/,
            '$1List<ReactPackage> packages = new PackageList(this).getPackages();\n$1packages.add(new OnnxruntimePackage());\n$1return packages;',
          );
        } else {
          config.modResults.contents = generateCode.mergeContents({
            src: config.modResults.contents,
            newSrc: '      packages.add(new OnnxruntimePackage());',
            tag: 'onnxruntime-react-native-package',
            anchor: /^\s*List<ReactPackage>\s+packages\s*=\s*new PackageList\(this\)\.getPackages\(\);\s*$/m,
            offset: 1,
            comment: '//',
          }).contents;
        }
      }
    }
    return config;
  });

  // Add build dependency to pod file
  config = configPlugin.withDangerousMod(config, [
    'ios',
    (config) => {
      const podFilePath = path.join(config.modRequest.platformProjectRoot, 'Podfile');
      const contents = fs.readFileSync(podFilePath, { encoding: 'utf-8' });
      const packageRoot = path.dirname(
        require.resolve(`${pkg.name}/package.json`, {
          paths: [config.modRequest.projectRoot],
        })
      );
      const podPath = toPodfileRelativePath(config.modRequest.platformProjectRoot, packageRoot);
      const updatedContents = generateCode.mergeContents({
        src: contents,
        newSrc: `  pod 'onnxruntime-react-native', :path => '${podPath}'`,
        tag: 'onnxruntime-react-native',
        anchor: /^target.+do$/,
        offset: 1,
        comment: '  # onnxruntime-react-native',
      }).contents;
      fs.writeFileSync(podFilePath, updatedContents);
      return config;
    },
  ]);

  return config;
};

exports.default = configPlugin.createRunOncePlugin(withOrt, pkg.name, pkg.version);
