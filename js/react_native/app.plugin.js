const configPlugin = require('@expo/config-plugins');
const generateCode = require('@expo/config-plugins/build/utils/generateCode');
const pkg = require('onnxruntime-react-native/package.json');
const path = require('path');
const fs = require('fs');

const withOrt = (config) => {
  // Add build dependency to gradle file
  config = configPlugin.withAppBuildGradle(config, (config) => {
    if (config.modResults.language === 'groovy') {
      config.modResults.contents = generateCode
                                       .mergeContents({
                                         src: config.modResults.contents,
                                         newSrc: '    implementation project(\':onnxruntime-react-native\')',
                                         tag: 'onnxruntime-react-native',
                                         anchor: /^dependencies[ \t]*\{$/,
                                         offset: 1,
                                         comment: '    // onnxruntime-react-native'
                                       })
                                       .contents;
    } else {
      throw new Error('Cannot add ONNX Runtime maven gradle because the build.gradle is not groovy');
    }

    return config;
  });

  // Add build dependency to pod file
  config = configPlugin.withDangerousMod(config, [
    'ios',
    (config) => {
      const podFilePath = path.join(config.modRequest.platformProjectRoot, 'Podfile');
      const contents = fs.readFileSync(podFilePath, {encoding: 'utf-8'});
      const updatedContents =
          generateCode
              .mergeContents({
                src: contents,
                newSrc: '  pod \'onnxruntime-react-native\', :path => \'../node_modules/onnxruntime-react-native\'',
                tag: 'onnxruntime-react-native',
                anchor: /^target.+do$/,
                offset: 1,
                comment: '  # onnxruntime-react-native'
              })
              .contents;
      fs.writeFileSync(podFilePath, updatedContents);
      return config;
    }
  ]);

  return config;
};

exports.default = configPlugin.createRunOncePlugin(withOrt, pkg.name, pkg.version)
