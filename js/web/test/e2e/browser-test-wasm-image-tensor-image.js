// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const IMAGE_HEIGHT = 50
const IMAGE_WIDTH = 50

function getRndColor() {
  let r = 255*Math.random()|0,
      g = 255*Math.random()|0,
      b = 255*Math.random()|0;
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

it('Browser E2E testing - Tensor <--> Image E2E test', async function () {

  // Creating Image HTML Image Element
  let img = new Image();
  img.crossOrigin = 'Anonymous';

  // Creating canvas element
  const canvas = document.createElement('canvas');
  canvas.height = IMAGE_HEIGHT;
  canvas.width = IMAGE_WIDTH;
  const context = canvas.getContext('2d');
  let y, x;

  // Filling the canvas with random data
  for(y = 0; y < IMAGE_HEIGHT; y++) {
    for(x = 0; x < IMAGE_WIDTH; x++) {
        context.fillStyle = getRndColor();
        context.fillRect(x, y, 1, 1);
      }
  }

  // Copying the canavas data to the image
  img.src = canvas.toDataURL();

  // Testing HTML Image Element --> Tensor --> ImageData --> Tensor
  img.onload = async () =>{
    // Image HTML element to tensor API - HTML
    const inputTensorHTML = await ort.Tensor.fromImage(img);
    // Tensor to ImageDAta API
    let newImage = inputTensorHTML.toImageData();
    // ImageData to tensor API
    let inputTensorImageData = await ort.Tensor.fromImage(newImage);

    for (let i = 0; i < newImage.height*newImage.width*3; i++) {
      if(inputTensorImageData.data[i]!==inputTensorHTML.data[i]){
        console.log("Element - " + i + " - " + inputTensorHTML.data[i] + " - " + inputTensorImageData.data[i]);
        throw new Error('BUG in HTML image element & ImageData use case');
      }
    }
  };

  // Copying the canavas data to the image as Data URL
  let image = canvas.toDataURL();

  // Testing Data URL --> Tensor --> Data URL --> Tensor
  // Data URL to tensor API -
  const inputTensorDataURL = await ort.Tensor.fromImage(image);
  // Tensor to ImageDAta API
  let newImage = inputTensorDataURL.toDataURL();
  // ImageData to tensor API
  let inputTensorImageData = await ort.Tensor.fromImage(newImage);

  for (let i = 0; i < newImage.height*newImage.width*3; i++) {
    if(inputTensorImageData.data[i]!==inputTensorDataURL.data[i]){
      console.log("Element - " + i + " - " + inputTensorHTML.data[i] + " - " + inputTensorImageData.data[i]);
      throw new Error('BUG in ImageData & Data URL use case');
    }
  }

  // Testing URL --> Tensor --> ImageData --> Tensor
  let online = navigator.onLine;
  if(online){
    // URL element to tensor API
    const inputTensorURL = await ort.Tensor.fromImage('https://media.istockphoto.com/id/172859087/photo/square-eggs.jpg?s=2048x2048&w=is&k=20&c=KiBRyyYaoUUSjcJLBh1-qqVu7LW6UQZBopZdva0f5e4=');
    // Tensor to ImageDAta API
    let newImage = inputTensorURL.toImageData();
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
