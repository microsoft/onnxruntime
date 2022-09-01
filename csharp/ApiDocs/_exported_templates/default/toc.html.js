// Copyright (c) Microsoft. All rights reserved. Licensed under the MIT license. See LICENSE file in the project root for full license information.
var extension = require('./toc.extension.js')

exports.transform = function (model) {

  if (extension && extension.preTransform) {
    model = extension.preTransform(model);
  }

  transformItem(model, 1);
  if (model.items && model.items.length > 0) model.leaf = false;
  model.title = "Table of Content";
  model._disableToc = true;

  if (extension && extension.postTransform) {
    model = extension.postTransform(model);
  }

  return model;

  function transformItem(item, level) {
    // set to null incase mustache looks up
    item.topicHref = item.topicHref || null;
    item.tocHref = item.tocHref || null;
    item.name = item.name || null;

    item.level = level;
    if (item.items && item.items.length > 0) {
      var length = item.items.length;
      for (var i = 0; i < length; i++) {
        transformItem(item.items[i], level + 1);
      };
    } else {
      item.items = [];
      item.leaf = true;
    }
  }
}
