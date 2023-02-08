---
title: Mobile super resolution
description: Build Android and iOS applications for improving image resolution with built-in pre and post processing
parent: Deploy on mobile
grand_parent: Tutorials
nav_order: 3
---

# Machine learning mobile application to improve image resolution

Learn how to build an application to improve image resolution using ONNX Runtime Mobile, with a model that includes pre and post processing.

You can use this tutorial to build the application for Android or iOS.

## Prepare the model

The machine learning model used in this tutorial is based on the one used in the PyTorch tutorial referenced at the bottom of this page.

We provide a convenient Python script that exports the PyTorch model into ONNX format and adds pre and post processing.

To run this script, install the following python packages first.

```bash
pip install pytorch
pip install onnxruntime
pip install onnxruntime-extensions
pip install pillow
python -m onnxruntime-extensions.tools superresolution_e2e.py
```

After the script runs, you should see two ONNX files in the folder where you ran the script.

```bash
pytorch_superresolution.onnx
pytorch_superresolution_with_pre_and_post_proceessing.onnx
```

If you load the second model into netron you can see its inputs and outputs: both image bytes.

Now it's time to write the application code.

## Android app

### Pre-requisites

* Android Studio Dolphin | 2021.3.1 Patch + (installed on Mac/Windows/Linux)
* Android SDK 29+
* Android NDK r22+
* An Android device or an Android Emulator

### Sample code

You can find full [source code for the Android super resolution app](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/super_resolution/android) in GitHub.

To run the app from source code, clone the above repo and load the android project into Android studio, build and run!

To build the app, step by step, follow the following sections.

### Code from scratch

#### Setup project

Create a new project for Phone and Tablet in Android studio and select the blank template. Call the application `super_resolution` or similar.

#### Dependencies

Add the following dependencies to the app `build.gradle`:

```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:latest.release'
// TODO: update with released version aar package when available
implementation files('libs/onnxruntime-extensions-android-0.6.0.aar')
```

#### Project resources

1. Add the model file as a raw resource

   Create a folder called `raw` in the `resources` folder and move or copy the ONNX model into the raw folder.

2. Add the test image as an asset

   Create a folder called `assets` in the main project folder and copy the image that you want to run super resolution on into that folder with the filename of `test_superresolution.png`

#### Main application class code

Create a file called MainActivity.kt and add the following pieces of code to it.

1. Add import statements

   ```kotlin
   import ai.onnxruntime.*
   import ai.onnxruntime.extensions.OrtxPackage
   import android.annotation.SuppressLint
   import android.os.Bundle
   import android.widget.Button
   import android.widget.ImageView
   import android.widget.Toast
   import androidx.activity.*
   import androidx.appcompat.app.AppCompatActivity
   import kotlinx.android.synthetic.main.activity_main.*
   import kotlinx.coroutines.*
   import java.io.InputStream
   import java.util.*
   import java.util.concurrent.ExecutorService
   import java.util.concurrent.Executors
   ```

2. Add the class variables for the main activity class

   ```kotlin
   private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
   private var outputImage: ImageView? = null
   private var superResolutionButton: Button? = null
   ```

3. Add the onCreate method

   ```kotlin
   override fun onCreate(savedInstanceState: Bundle?) {
       super.onCreate(savedInstanceState)
       setContentView(R.layout.activity_main)

       outputImage = findViewById(R.id.imageView2);
       superResolutionButton = findViewById(R.id.super_resolution_button)

       superResolutionButton?.setOnClickListener {
           performSuperResolution()
               Toast.makeText(baseContext, "Super resolution performed!", Toast.LENGTH_SHORT).show()
        }
    }
   ```

4. Add the onDestroy method

   ```kotlin
   override fun onDestroy() {
       super.onDestroy()
       ortEnv.close()
   }
   ```

5. Add the updateUI method

   ```kotlin
   private fun updateUI(result: Result) {
   outputImage?.setImageBitmap(result.outputBitmap)
   }
   ```

6. Add the readModel method

   This method reads the ORT format model from the resources folder.

   ```kotlin
   private fun readModel(): ByteArray {
       val modelID = R.raw.pt_super_resolution_op16
       return resources.openRawResource(modelID).readBytes()
   }   
   ```

7. Add the readInputImage method

   This method reads a test image from the assets folder. Currently it reads a fixed image built into the application. The sample will soon be extended to read the image directly from the camera or the camera roll.

   ```kotlin
   private fun readInputImage(): InputStream {
       return assets.open("test_superresolution.png")
   }   
   ```

8. Add a method to create the ONNX Runtime session

   A session holds a reference to the model used to perform inference in the application. It also takes a session options parameter, which is where you can specify different execution providers (hardware accelerators such as NNAPI). In this case, we default to running on CPU. We do however register the custom op library where the image encoding and decoding operators at the input and output of the model are found.

   ```kotlin
   private fun createOrtSession(): OrtSession {
       val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
       sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
       return ortEnv.createSession(readModel(), sessionOptions)
   }
   ```

9. Add the method to perform inference

   This method is calls the heart of the application: `SuperResPerformer.upscale()`, which is the method that runs inference on the model. The code for this is shown in the next section.

   ```kotlin
   private fun performSuperResolution() {
       var superResPerformer = SuperResPerformer(createOrtSession())
       var result = superResPerformer.upscale(readInputImage(), ortEnv)
       updateUI(result);
   }   
   ```

10. Add the TAG object

   ```kotlin
   companion object {
       const val TAG = "ORTSuperResolution"
   }
   ```

#### Model inference class code

Create a file called `SuperResPerformer.kt` and add the following snippets of code to it.

1. Add imports

   ```kotlin
   import ai.onnxruntime.OnnxJavaType
   import ai.onnxruntime.OrtSession
   import ai.onnxruntime.OnnxTensor
   import ai.onnxruntime.OrtEnvironment
   import android.graphics.Bitmap
   import android.graphics.BitmapFactory
   import java.io.InputStream
   import java.nio.ByteBuffer
   import java.util.*
   ```

2. Create a result class

   ```kotlin
   internal data class Result(
       var outputBitmap: Bitmap? = null
   ) {}
   ```

3. Create the super resolution performer class

   This class and its main function `upscale` are where most of the calls to ONNX Runtime live.

   * The `OrtEnvironment` singleton maintains properties of the environment and configured logging levels
   * `OnnxTensor.createTensor()` is used to create a tensor made up of the input image bytes, suitable as input to the model
   * `OnnxJavaType.UINT8` is the data type of the input tensor
   * `OrtSession.run()` run the inference (prediction) on the model to get the output upscaled image

   ```kotlin
   internal class SuperResPerformer(
    private val ortSession: OrtSession
   ) {

       fun upscale(inputStream: InputStream, ortEnv: OrtEnvironment): Result {
           var result = Result()

           // Step 1: convert image into byte array (raw image bytes)
           val rawImageBytes = inputStream.readBytes()

           // Step 2: get the shape of the byte array and make ort tensor
           val shape = longArrayOf(rawImageBytes.size.toLong())

           ortEnv.use {
               val inputTensor = OnnxTensor.createTensor(
                   ortEnv,
                   ByteBuffer.wrap(rawImageBytes),
                   shape,
                   OnnxJavaType.UINT8
               )
               inputTensor.use {
                   // Step 3: call ort inferenceSession run
                   val output = ortSession.run(Collections.singletonMap("image", inputTensor))

                   // Step 4: output analysis
                   output.use {
                       val rawOutput = (output?.get(0)?.value) as ByteArray
                       val outputImageBitmap = byteArrayToBitmap(rawOutput)

                       // Step 5: set output result
                       result.outputBitmap = outputImageBitmap
                   }
               }
           }
           return result
       }

       private fun byteArrayToBitmap(data: ByteArray): Bitmap {
           return BitmapFactory.decodeByteArray(data, 0, data.size)
       }

       protected fun finalize() {
           ortSession.close()
       }
   }
   ```

### Build and run the app

Within Android studio:

* Select Build -> Make Project
* Run -> app

The app runs in the device emulator. Connect to your Android device to run the app on device.

## Resources

[Original PyTorch tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
