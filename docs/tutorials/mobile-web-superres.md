# Super Resolution on Android and Web

Super resolution model improves the resolution of a low-resolution image. This tutorial contains an application capable of performing super-resolution on input images using ONNX Runtime and the application can be run on both Mobile and Web platforms. The IOS implementation will be added soon.

This project makes use of the [Expo framework](https://docs.expo.dev/) which is a free and open source toolchain built around React Native to help you build native projects using JavaScript and React.

**NOTE:** `This application makes use of a Web API that may not be supported in some browsers. Check Browser Compatibility` [here](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas#browser_compatibility)

The ORT format model used can be found [here](https://github.com/VictorIyke/super_resolution_MW/blob/main/cross_plat/assets/super_resnet12.ort).

## Contents
1. Pre-requisites
2. Model
3. Pre-processing
4. Post-processing


## Pre-requisites
1. Install Node.js

2. Install Java (OpenJDK).
   - Download and install [OpenJDK Version 11](https://adoptopenjdk.net/).

   - Set JAVA_HOME environment variable. Steps are highlighted [here](https://java2blog.com/how-to-set-java-path-windows-10/#How_to_set_JAVA_HOME_in_Windows_10).

3. Install the [expo-cli](https://docs.expo.dev/)

  ```sh
  npm install -g expo-cli
  ```

4. Install yarn
  ```sh
  npm install -g yarn
  ```

For steps on how to build this project, refer to [this](https://github.com/VictorIyke/onnxruntime-inference-examples/blob/super_res/js/app-super-resolution/instructions.md).


## Model
For this example, the model used will be in [ORT format](https://onnxruntime.ai/docs/reference/ort-format-models.html#what-is-the-ort-model-format). This format is usually used in size-constrained environments hence why it is used here.

The super resolution model accepts a Float32 array of a 224 x 224 image as an input and provides the image as a 672 x 672 Float32 array as an output. This model also operates on the Y channel of YCbCr.

## Pre-processing
 To pre-process the image, the image has to be resized/cropped, the pixel data from the image has to be obtained, converted from the RGB format to the YCbCr format, and put into a Float32 ORT tensor. The process of obtaining the pixel data is done differently between the mobile and web.

- ### Mobile

    In React Native, there isn't a direct way to obtain the pixel data from an image, however there is a feature called [Android Native Modules](https://reactnative.dev/docs/next/native-modules-intro). This feature allows React Native to access Java APIs that are needed to grab the pixel data from an image. 

    As shown below, the Java method takes in the file path of the image, produces the pixel data of the image, and sends the pixel data back to React Native as a promise. This can be found in the `BitmapModule.java` file.

    ```java
    // Name of the Native Module to be called in JavaScript
    public String getName() {
      return "Bitmap";
    }
    /**
     * It provides an object containing the pixel data, width, height, and hasAlpha boolean value of an image given its source (filePath).
     * The filePath must be a string and the object is returned to JavaScript as a promise.
     */
    @ReactMethod
    public void getPixels(String filePath, final Promise promise) {
        try {
            WritableNativeMap result = new WritableNativeMap();
            WritableNativeArray pixels = new WritableNativeArray();
            ...
            // Java Logic
            ...
            result.putArray("pixels", pixels);
            promise.resolve(result);
        } catch (Exception e) {
            promise.reject(e);
        }
    }
    ```
    Here in our project, the Native Module `Bitmap` created in the `BitmapModule.java` file is called and the `getPixels` method is used to obtain the pixel data of the selected image.
    ```js
    `mobile.tsx`

    const imageDim = 224
    ...
    let bitmapPixel: number[] = Array(imgDim*imgDim);

    const bitmapModule = NativeModules.Bitmap
    ...
    
    bitmapPixel = await bitmapModule.getPixels(imageResult.uri).then(

      (image: any) => {
        return Array.from(image.pixels);
      }
    )
    ```

- ## Web

  The Native Modules feature is usless for the web as it is used to access Java APIs. Instead, the [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) can be used to get the pixel data from an image through the web. 

  Similar to the Canvas API, there's another Web API called an [Offscreen Canvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas), which can be used to draw images in a canvas that the user can't see. A context has to provided in order to use these canvas. Once a context exist, an image can be drawn on that Offscreen canvas. We do the same thing for a normal canvas that will be used later.

  The pixel data can be obtained from the drawn image on the Offscreen canvas and put into an array. The data is already arranged in RGB format, so the binary operations that were used for mobile are not needed here. To avoid complications with having too many images being put onto the Offscreen canvas, the image is cleared after the image data is stored.

  ```js
  `web.tsx`

  let kdv: OffscreenCanvasRenderingContext2D | null
  let offscreen: any 
  if (platform == "web") {offscreen = new OffscreenCanvas(1000, 1000)}
  ...
  async function draw() {
      const image1 = document.getElementById('selectedImage') as HTMLImageElement
      kdv = offscreen.getContext('2d')

      if (kdv) {
        kdv.drawImage(image1, 0, 0, imageDim, imageDim)  
        const myImageData = kdv.getImageData(0, 0, imageDim, imageDim)
        bitmapPixel = Array.from(myImageData.data)
        kdv.clearRect(0, 0, imageDim, imageDim)
      }
  }
  ```

The pixel data that is received is in different formats based on the platform. For Mobile, the RGBA values of each pixel are merged to single hex value while for web, the RGBA values are already separated. After obtaining the RGB values, the values are then converted into YCbCr format.

```js
`utilities.ts`

export async function converter(array: number[][], mode: "YCbCR"|"RGB", platform: PlatformTypes) {
  ...
  for (let i = 0; i < imageDim*imageDim; i++) {

    let red = 0
    let green = 0
    let blue = 0
    if (platform == "android") {
      const value = inputArray[i] // inputArray contains the pixel data
      red = (value >> 16 & 0xFF)
      green = (value >> 8 & 0xFF)
      blue = (value & 0xFF)
      } 
    else if(platform == "web") {
      const currIndex = i * 4;
      red = inputArray[currIndex]
      green = inputArray[currIndex + 1]
      blue = inputArray[currIndex + 2]
    };

    floatPixelsY[i] = pixelsRGBToYCbCr(red, green, blue, "y")
    ...
    cbArray[i] = pixelsRGBToYCbCr(red, green, blue, "cb")
    crArray[i] = pixelsRGBToYCbCr(red, green, blue, "cr")
  }
  ...
}
...
function pixelsRGBToYCbCr(red: number, green: number, blue: number, mode: string) {
  let result = 0
  if (mode == "y") {
      result  = (0.299 * red +
                  0.587 * green +
                  0.114 * blue) / 255
  }else if (mode == "cb"){
      result = ((-0.168935) * red +
                  (-0.331665) * green +
                  0.50059 * blue) + 128
  }else if (mode == "cr") {
      result = ((0.499813 * red +
          (-0.418531) * green +
          (-0.081282) * blue) + 128)
  }
  return result

}
```

The Y, Cb, and Cr channels are then seperated and put into arrays. The Y' array is put into an ORT tensor and fed to the model. 


## Post-processing
The model's output is a 672 x 672 array that is used for the post-processing. During the post-prcessing, the output array from the model needs to matched with Cb and Cr values, converted to RGB format, and displayed to the user.

The Y, Cb, and Cr channels are converted back into the RGB format using some pixel conversions and those RGB values are stored in an array.

```js
`utilities.ts`

function pixelsYCbCrToRGB(pixel: number, cb: number, cr: number, platform: PlatformTypes) {
    const y = Math.min(Math.max((pixel * 255), 0), 255);

    const red = Math.min(Math.max((y + (1.4025 * (cr-0x80))), 0), 255);

    const green = Math.min(Math.max((y + ((-0.34373) * (cb-0x80)) +
                                          ((-0.7144) * (cr-0x80))), 0), 255);

    const blue = Math.min(Math.max((y + (1.77200 * (cb-0x80))), 0), 255);
}
...
... 
export async function converter(array: number[][], mode: "YCbCR"|"RGB", platform: PlatformTypes) {
  ...
  let intArray = platform == "android"? new Array(scaledDim*scaledDim): new Array(scaledDim*scaledDim*4)

  for (let i=0; i < scaledDim*scaledDim; i++) {
      if (platform == "android") {
          intArray[i] = pixelsYCbCrToRGB(outputArray[i], cbArray[i], crArray[i], platform)[0]

      }else if (platform == "web") {
          const pixel = pixelsYCbCrToRGB(outputArray[i], cbArray[i], crArray[i], platform)
          const currIndex = i * 4;

          intArray[currIndex] = pixel[0]
          intArray[currIndex + 1] = pixel[1]
          intArray[currIndex + 2] = pixel[2]
          intArray[currIndex + 3] = 255
      }
  }
  return intArray
  ...
}
...
```

Now that the pixels are back into RGB format, the output image can be displayed to the user. Depending on the platform, there are different ways to do that.  
- ### Mobile

  After the processed output data is obtained, the `Bitmap` Native Module is called again. The module has a `getImageUri` method that takes in pixel data, creates a bitmap image from the pixel data, store the bitmap in a temporary file, and sends the file uri back to React Native as a promise. This can also be found in the `BitmapModule.java` file.

  ```java
  // Name of the Native Module to be called in JavaScript
  public String getName() {
    return "Bitmap";
  }
  
  ...
  /**
   * It provides an image source given a pixel data.
   * Using the pixel data, an image is generated by creating a bitmap object and saving it
   * in a temporary location on the device.
   * Then the path of the temporary image file is sent back to JavaScript as a promise.
   */
  @ReactMethod
  public void getImageUri(ReadableArray arrayPixels, final Promise promise) {
      try {
          WritableNativeMap result = new WritableNativeMap();
          int[] arrayIntPixels = new int[224*224*3*3];
          ...
          Bitmap bitmap = Bitmap.createBitmap(672, 672, Bitmap.Config.ARGB_8888);
          bitmap.copyPixelsFromBuffer(IntBuffer.wrap(arrayIntPixels));
          ...
          File tempFile = new File(getReactApplicationContext().getExternalFilesDir(null), "HROutput.jpg");
          ...
          result.putString("uri", tempFile.toString());
          promise.resolve(result);
      } catch (Exception e) {
          promise.reject(e);
      }
    }
  ```
  Here in our project, `getImageUri` method is used to obtain the image source of the output image and the output image is set to be displayed.
  ```js
  `mobile.tsx`

  /**
   * It sets the output image visible by generating the output image source given its pixel data.
   * Makes use of an Android [Native Module](https://reactnative.dev/docs/next/native-modules-android) which creates a temporary image file
   * from its pixel data
   */
  async function postprocess(floatArray: number[]) {

    const intArray = await converter([floatArray, Array.from(cbArray), Array.from(crArray)], "RGB", platform) as any[]

    let imageUri = await bitmapModule.getImageUri(Array.from(intArray)).then(
      (image: any) => {
        return image.uri
      }
    )

    const imageRotated = await ImageManipulator.manipulateAsync(imageUri, [
      { rotate: 90 },
      { flip: ImageManipulator.FlipType.Horizontal }
    ])

    setOutputImage({ localUri: imageRotated.uri })
  };
  ...

  return(
    ...
    <Image
      source={{ uri: outputImage.localUri }}
      style={styles.thumbnail}
    />
    ...
  )
  ```

- ### Web

  The main canvas is created in the App return function so that the user can see the output image. The processed output data is drawn onto the Offscreen canvas, resized and then transferred to the main canvas.

  ```js
  `web.tsx`
  let kdv: OffscreenCanvasRenderingContext2D | null
  let ctx: CanvasRenderingContext2D |null
  ...
  export default function WebApp({navigation, route}: MainScreenProps) {

    ...
    async function postProcess(outputArray: number []) {
        const newImageData = await converter(Array.of(outputArray, Array.from(cbArray), Array.from(crArray)), "RGB", platform) as any[]
        let data = myImageScaledData.data // myImageScaledData is an ImageData object useful for canvas API

        newImageData.forEach((value, index) => {data[index] = value})
        
        if (ctx && kdv) {  
          kdv.putImageData(myImageScaledData, 0, 0);
          kdv.save();
          ctx.drawImage(offscreen, 0, 0, scaledImageDim, scaledImageDim, 0, 0, 350, 350)
          ctx.save()
        }
      }

    ...
    return(

      ...
      <canvas id='canvas' width="350" height="350">
          <img id='selectedImage' src={selectedImage.localUri} width="250" height="250" alt='' />
      </canvas>
      ...
    )
  }
  ```
































