// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - Tensor <--> Image E2E test', async function () {
  // Image URL to tensor API
  const inputTensor = await ort.Tensor.fromImage('https://onnxruntime.ai/images/ONNX-Icon.png');
  // Tensor to ImageDAta API
  const newImage = inputTensor.toImageData();
  // ImageData to tensor API
  const inputTensor2 = await ort.Tensor.fromImage(newImage);

  for (let i = 0; i < newImage.height*newImage.width*3; i++) {
    if(inputTensor.data[i]!==inputTensor2.data[i]){
      throw new Error('BUG in ImageData & URL');
    }
  }
});
