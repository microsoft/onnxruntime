module.exports = {
  dependency: {
    platforms: {
      android: {
        packageImportPath: 'import ai.onnxruntime.reactnative.OnnxruntimePackage;',
        packageInstance: 'new OnnxruntimePackage()',
      },
      ios: {},
    },
  },
};
