'use strict';

const path = require('path');
<<<<<<< HEAD
const exclustionlist = require('metro-config/src/defaults/exclusionList');
=======
const exclusionList = require('metro-config/src/defaults/exclusionList');
>>>>>>> 97b8f6f394ae02c73ed775f456fd85639c91ced1
const escape = require('escape-string-regexp');
const pak = require('../package.json');

const root = path.resolve(__dirname, '..');

const modules = Object.keys({
  ...pak.peerDependencies,
});

module.exports = {
  projectRoot: __dirname,
  watchFolders: [root],

  // We need to make sure that only one version is loaded for peerDependencies
<<<<<<< HEAD
  // So we exclustionlist them at the root, and alias them to the versions in example's node_modules
  resolver: {
    exclustionlistRE: exclustionlist(
=======
  // So we exclusionList them at the root, and alias them to the versions in example's node_modules
  resolver: {
    exclusionListRE: exclusionList(
>>>>>>> 97b8f6f394ae02c73ed775f456fd85639c91ced1
      modules.map(
        (m) =>
          new RegExp(`^${escape(path.join(root, 'node_modules', m))}\\/.*$`)
      )
    ),

    extraNodeModules: modules.reduce((acc, name) => {
      acc[name] = path.join(__dirname, 'node_modules', name);
      return acc;
    }, {}),
  },

  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: true,
      },
    }),
  },
};
