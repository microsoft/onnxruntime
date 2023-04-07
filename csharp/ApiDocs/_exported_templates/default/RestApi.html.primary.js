// Copyright (c) Microsoft. All rights reserved. Licensed under the MIT license. See LICENSE file in the project root for full license information.

var restApiCommon = require('./RestApi.common.js');
var extension = require('./RestApi.extension.js')

exports.transform = function (model) {
  if (extension && extension.preTransform) {
    model = extension.preTransform(model);
  }

  if (restApiCommon && restApiCommon.transform) {
    model = restApiCommon.transform(model);
  }
  model._disableToc = model._disableToc || !model._tocPath || (model._navPath === model._tocPath);

  if (extension && extension.postTransform) {
    model = extension.postTransform(model);
  }

  return model;
}

exports.getOptions = function (model) {
  return { "bookmarks": restApiCommon.getBookmarks(model) };
}