// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const IMAGE_HEIGHT = 20
const IMAGE_WIDTH = 15

function getRndColor() {
  let r = 255*Math.random()|0,
      g = 255*Math.random()|0,
      b = 255*Math.random()|0,
      a = 255*Math.random()|0;
  return 'rgb(' + r + ',' + g + ',' + b + ',' + a +')';
}

function compareTensors(tensorA, tensorB, msg){
  for (let i = 0; i < IMAGE_HEIGHT*IMAGE_WIDTH*3; i++) {
    if(tensorA.data[i]!==tensorB.data[i]){
      console.log("Element - " + i + " - " + tensorA.data[i] + " - " + tensorB.data[i]);
      throw new Error(msg);
    }
  }
}

// TODO: this testing need to be revised
//
// work item list:
// - format the code
// - remove 'wasm' from file name
// - test depending on public website (https://media.istockphoto.com/) should be changed to depends on a localhost
// - the test is composed by 3 different test cases. split them to 3 different cases.
// - some test cases are wriiten incorrectly.
//
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
    const inputTensorHTML = await ort.Tensor.fromImage(img, {norm:{bias:[2,3,9,5],mean:[5,6,17,8]}});
    // Tensor to ImageDAta API
    let newImage = inputTensorHTML.toImageData({norm:{bias:[2/5,3/6,9/17,5/8],mean:[5,6,17,8]}});
    // ImageData to tensor API
    let inputTensorImageData = await ort.Tensor.fromImage(newImage, options={norm:{bias:[2,3,9,5],mean:[5,6,17,8]}});

    // TODO: fix this test case
    //
    // the line above does not return as expected because syntax error.
    // the reason why it does not fail is because it throws exception, and the exception is not caught. the line below is not executed.
    // to fix this, wrap a try-catch to deal with exceptions.

    compareTensors(inputTensorHTML,inputTensorImageData,'BUG in HTML image element & ImageData use case');
  }

  // Copying the canavas data to the image as Data URL
  let image = canvas.toDataURL();

  // Testing Data URL --> Tensor --> Data URL --> Tensor
  // Data URL to tensor API -
  const inputTensorDataURL = await ort.Tensor.fromImage(image,{format:'RBG', norm:{bias:[1,10,5,0],mean:[5,7,11,0]}});
  // Tensor to ImageDAta API
  let newImage = inputTensorDataURL.toDataURL({norm:{bias:[1/5,10/7,5/11,0],mean:[5,7,11,0]}});
  // ImageData to tensor API
  let inputTensorImageData = await ort.Tensor.fromImage(newImage,{format:'RGBA', norm:{bias:[1,10,5,0],mean:[5,7,11,0]}});

  // TODO: fix this
  // creating tensor from image data should not depend on `options.format`.
  // data url with type 'image/png' has a determined 'RGBA' format

  compareTensors(inputTensorDataURL,inputTensorImageData,'BUG in ImageData & Data URL use case');

  // Testing URL --> Tensor --> ImageData --> Tensor
  let online = navigator.onLine;
  if(online){
    // URL element to tensor API
    const inputTensorURL = await ort.Tensor.fromImage('https://media.istockphoto.com/id/172859087/photo/square-eggs.jpg?s=2048x2048&w=is&k=20&c=KiBRyyYaoUUSjcJLBh1-qqVu7LW6UQZBopZdva0f5e4=',{norm:{bias:[2,3,9,0],mean:[5,6,17,0]}});
    // Tensor to ImageDAta API
    let newImage = inputTensorURL.toImageData({format:'RGB',norm:{bias:[2/5,3/6,9/17,0],mean:[5,6,17,0]}});
    // ImageData to tensor API
    let inputTensorImageData = await ort.Tensor.fromImage(newImage,{format:'RGB',norm:{bias:[2,3,9,0],mean:[5,6,17,0]}});

    compareTensors(inputTensorURL,inputTensorImageData,'BUG in ImageData & URL');
  }else{
    console.log("No internet connection - didn't test Image URL to tensor API");
  }
});
