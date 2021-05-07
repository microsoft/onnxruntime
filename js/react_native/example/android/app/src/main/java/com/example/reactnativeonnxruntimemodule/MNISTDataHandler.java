// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package com.example.reactnativeonnxruntimemodule;

import static java.util.stream.Collectors.joining;

import ai.onnxruntime.reactnative.OnnxruntimeModule;
import ai.onnxruntime.reactnative.TensorHelper;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.util.Base64;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.math.MathUtils;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@RequiresApi(api = Build.VERSION_CODES.N)
public class MNISTDataHandler extends ReactContextBaseJavaModule {
  private static ReactApplicationContext reactContext;

  MNISTDataHandler(ReactApplicationContext context) throws Exception {
    super(context);
    reactContext = context;
  }

  @NonNull
  @Override
  public String getName() {
    return "MNISTDataHandler";
  }

  // It returns mode path in app package,
  // so that onnxruntime is able to load a model using a given path.
  @ReactMethod
  public void getLocalModelPath(Promise promise) {
    try {
      String modelPath = copyFile(reactContext, "mnist.onnx");
      promise.resolve(modelPath);
    } catch (Exception e) {
      promise.reject("Can't get a mdoel", e);
    }
  }

  // It returns image path in app package.
  @ReactMethod
  public void getImagePath(Promise promise) {
    try {
      String imageUri = copyFile(reactContext, "3.jpg");
      promise.resolve(imageUri);
    } catch (Exception e) {
      promise.reject("Can't get a image", e);
    }
  }

  // It gets raw input data, which can be uri or byte array and others,
  // returns cooked data formatted as input of a model by promise.
  @ReactMethod
  public void preprocess(String uri, Promise promise) {
    try {
      WritableMap inputDataMap = preprocess(uri);
      promise.resolve(inputDataMap);
    } catch (Exception e) {
      promise.reject("Can't process an image", e);
    }
  }

  // It gets a result from onnxruntime and a duration of session time for input data,
  // returns output data formatted as React Native map by promise.
  @RequiresApi(api = Build.VERSION_CODES.KITKAT)
  @ReactMethod
  public void postprocess(ReadableMap result, Promise promise) {
    try {
      WritableMap cookedMap = postprocess(result);
      promise.resolve(cookedMap);
    } catch (Exception e) {
      promise.reject("Can't process a inference result", e);
    }
  }

  // It gets raw input data, which can be uri or byte array and others,
  // returns cooked data formatted as input of a model by promise.
  private WritableMap preprocess(String uri) throws Exception {
    final int batchSize = 1;
    final int channelSize = 1;
    final int imageHeight = 28;
    final int imageWidth = 28;

    InputStream is = MainApplication.getAppContext().getContentResolver().openInputStream(Uri.parse(uri));
    BufferedInputStream bis = new BufferedInputStream(is);
    byte[] imageArray = new byte[bis.available()];
    bis.read(imageArray);

    Bitmap bitmap = BitmapFactory.decodeByteArray(imageArray, 0, imageArray.length);
    if (bitmap == null) {
      throw new Exception("Can't decode image: " + uri);
    }
    // Resize bitmap to 28x28
    bitmap = Bitmap.createScaledBitmap(bitmap, imageWidth, imageHeight, false);

    ByteBuffer imageByteBuffer =
        ByteBuffer.allocate(imageHeight * imageWidth * channelSize * 4).order(ByteOrder.nativeOrder());
    FloatBuffer imageFloatBuffer = imageByteBuffer.asFloatBuffer();
    for (int h = 0; h < imageHeight; ++h) {
      for (int w = 0; w < imageWidth; ++w) {
        int pixel = bitmap.getPixel(w, h);
        imageFloatBuffer.put((float)Color.red(pixel));
      }
    }
    imageByteBuffer.rewind();

    WritableMap inputDataMap = Arguments.createMap();

    // dims
    WritableMap inputTensorMap = Arguments.createMap();

    WritableArray dims = Arguments.createArray();
    dims.pushInt(batchSize);
    dims.pushInt(channelSize);
    dims.pushInt(imageHeight);
    dims.pushInt(imageWidth);
    inputTensorMap.putArray("dims", dims);

    // type
    inputTensorMap.putString("type", TensorHelper.JsTensorTypeFloat);

    // data encoded as Base64
    imageByteBuffer.rewind();
    String data = Base64.encodeToString(imageByteBuffer.array(), Base64.DEFAULT);
    inputTensorMap.putString("data", data);

    inputDataMap.putMap("Input3", inputTensorMap);

    return inputDataMap;
  }

  @RequiresApi(api = Build.VERSION_CODES.KITKAT)
  private WritableMap postprocess(ReadableMap result) throws Exception {
    String detectionResult = "";

    ReadableMap outputTensor = result.getMap("Plus214_Output_0");

    String outputData = outputTensor.getString("data");
    FloatBuffer buffer =
        ByteBuffer.wrap(Base64.decode(outputData, Base64.DEFAULT)).order(ByteOrder.nativeOrder()).asFloatBuffer();
    ArrayList<Double> dataArray = new ArrayList<>();
    while (buffer.hasRemaining()) {
      dataArray.add((double)buffer.get());
    }

    final double max = Collections.max(dataArray);
    double total = 0.0f;
    for (int i = 0; i < dataArray.size(); ++i) {
      dataArray.set(i, Math.exp((double)dataArray.get(i) - max));
      total += dataArray.get(i);
    }
    double[] softmax = new double[dataArray.size()];
    int argmax = 0;
    double maxValue = 0;
    for (int i = 0; i < dataArray.size(); ++i) {
      softmax[i] = dataArray.get(i) / total;
      if (softmax[i] > maxValue) {
        maxValue = softmax[i];
        argmax = i;
      }
    }

    if (max == 0) {
      detectionResult = "No match";
    } else {
      detectionResult = "I guess, it's " + argmax;
    }

    WritableMap cookedMap = Arguments.createMap();
    cookedMap.putString("result", detectionResult);

    return cookedMap;
  }

  /*
    Copy a file from assets to data folder and return an uri for copied file.
   */
  private static String copyFile(Context context, String filename) throws Exception {
    File file = new File(context.getExternalFilesDir(null), filename);
    if (!file.exists()) {
      try (InputStream in = context.getAssets().open(filename)) {
        try (OutputStream out = new FileOutputStream(file)) {
          byte[] buffer = new byte[1024];
          int read = in.read(buffer);
          while (read != -1) {
            out.write(buffer, 0, read);
            read = in.read(buffer);
          }
        }
      }
    }

    return file.toURI().toString();
  }
}
