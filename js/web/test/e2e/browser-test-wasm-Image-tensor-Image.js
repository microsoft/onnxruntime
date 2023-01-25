// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

function getRndColor() {
  var r = 255*Math.random()|0,
      g = 255*Math.random()|0,
      b = 255*Math.random()|0;
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

it('Browser E2E testing - Tensor <--> Image E2E test', async function () {

  // Creating Image HTML Image Element
  var img = document.createElement('img');
  img.crossOrigin = 'Anonymous';
  var canvas = document.createElement('canvas');
  canvas.height = 200;
  canvas.width = 200;
  var context = canvas.getContext('2d');
  var y, x;

  for(y = 0; y < 200; y++) {
    for(x = 0; x < 200; x++) {
        context.fillStyle = getRndColor();
        context.fillRect(x, y, 1, 1);
      }
  }

  img = canvas.toDataURL();

  // Image HTML element to tensor API
  const inputTensorHTML = await ort.Tensor.fromImage(img);
  // Tensor to ImageDAta API
  var newImage = inputTensorHTML.toImageData();
  // ImageData to tensor API
  var inputTensorImageData = await ort.Tensor.fromImage(newImage);

  for (let i = 0; i < newImage.height*newImage.width*3; i++) {
    if(inputTensorImageData.data[i]!==inputTensorHTML.data[i]){
      console.log("Element - " + i + " - " + inputTensorHTML.data[i] + " - " + inputTensorImageData.data[i]);
      throw new Error('BUG in ImageData & URL');
    }
  }

  var online = navigator.onLine;

  if(online){
    // URL element to tensor API
    const inputTensorURL = await ort.Tensor.fromImage('https://media.istockphoto.com/id/172859087/photo/square-eggs.jpg?s=2048x2048&w=is&k=20&c=KiBRyyYaoUUSjcJLBh1-qqVu7LW6UQZBopZdva0f5e4=');
    // Tensor to ImageDAta API
    newImage = inputTensorURL.toImageData();
    // ImageData to tensor API
    inputTensorImageData = await ort.Tensor.fromImage(newImage);

    for (let i = 0; i < newImage.height*newImage.width*3; i++) {
      if(inputTensorURL.data[i]!==inputTensorImageData.data[i]){
        console.log("Element - " + i + " - " + inputTensorURL.data[i] + " - " + inputTensorImageData.data[i]);
        throw new Error('BUG in ImageData & URL');
      }
    }
  }else{
    console.log("No internet connection - didn't test Image URL to tensor API");
  }
});
