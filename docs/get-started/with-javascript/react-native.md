---
title: React Native
parent: JavaScript
grand_parent: Get Started
has_children: false
nav_order: 3
---

# Get started with ONNX Runtime for React Native

## Contents
{: .no_toc }

* TOC placeholder
{:toc}


## Install


```bash
# install latest release version
npm install onnxruntime-react-native
```

## Import


```js
// use ES6 style import syntax (recommended)
import * as ort from 'onnxruntime-react-native';
```
```js
// or use CommonJS style import syntax
const ort = require('onnxruntime-react-native');
```


### Enable ONNX Runtime Extensions for React Native
To enable support for [ONNX Runtime Extensions](https://github.com/microsoft/onnxruntime-extensions) in your React Native app,
you need to specify the following configuration as a top-level entry (note: usually where the package `name`and `version`fields are) in your project's root directory `package.json` file. 

```js
"onnxruntimeExtensionsEnabled": "true"
```

