// This should fix the warning about:
// Multiple Podfiles were found: ios/Podfile,e2e/ios/Podfile. Choosing ios/Podfile automatically.

module.exports = {
  project: {
    ios: {
      sourceDir: './ios',
    },
  },
};