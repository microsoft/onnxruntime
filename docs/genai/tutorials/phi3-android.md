---
title: Phi-3 for Android
description: Develop an Android generative AI application with ONNX Runtime
has_children: false
parent: Tutorials
grand_parent: Generate API (Preview)
nav_order: 1
---

# Build an Android generative AI application
This is a basic [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) Android example application using [ONNX Runtime mobile](https://onnxruntime.ai/docs/tutorials/mobile/) and [ONNX Runtime Generate() API](https://github.com/microsoft/onnxruntime-genai) with support for efficiently running generative AI models. This tutorial will walk you through how to download and run the Phi-3 App on your own mobile device so you can get started incorporating Phi-3 into your own mobile developments.  

## Capabilities
[Phi-3 Mini-4k-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) is a small language model used for language understanding, math, code, long context, logical reasoning, and more showcasing a robust and state-of-the-art performance among models with less than 13 billion parameters.

## Important Features

### Java API
This app uses the [generate() Java API's](https://github.com/microsoft/onnxruntime-genai/tree/main/src/java/src/main/java/ai/onnxruntime/genai) GenAIException, Generator, GeneratorParams, Model, and TokenizerStream classes ([documentation](https://onnxruntime.ai/docs/genai/api/java.html)). The [generate() C API](https://onnxruntime.ai/docs/genai/api/c.html), [generate() C# API](https://onnxruntime.ai/docs/genai/api/csharp.html), and [generate() Python API](https://onnxruntime.ai/docs/genai/api/python.html) are also available.

### Model Downloads
This app downloads the [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model through Hugging Face. To use a different model, change the path links to refer to your chosen model.
```java
final String baseUrl = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/";
List<String> files = Arrays.asList(
    "added_tokens.json",
    "config.json",
    "configuration_phi3.py",
    "genai_config.json",
    "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
    "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json");
```
These packages will only need to be downloaded once. While editing your app and running new versions, the downloads will skip since all files already exist.
```java
if (urlFilePairs.isEmpty()) {
    // Display a message using Toast
    Toast.makeText(this, "All files already exist. Skipping download.", Toast.LENGTH_SHORT).show();
    Log.d(TAG, "All files already exist. Skipping download.");
    model = new Model(getFilesDir().getPath());
    tokenizer = model.createTokenizer();
    return;
}
```
### Crash Prevention
Downloading the packages for the app on your mobile device takes ~15-30 minutes depending on which device you are using. The progress bar indicates what percent of the downloads are completed. 
```java
public void onProgress(long lastBytesRead, long bytesRead, long bytesTotal) {
    long lastPctDone = 100 * lastBytesRead / bytesTotal;
    long pctDone = 100 * bytesRead / bytesTotal;
    if (pctDone > lastPctDone) {}
        Log.d(TAG, "Downloading files: " + pctDone + "%");
        runOnUiThread(() -> {
            progressText.setText("Downloading: " + pctDone + "%");
        });
    }
}
```
Because the app is initialized when downloads start, the 'send' button for prompts is disabled until downloads are complete to prevent crashing.
```java
if (model == null) {
    // if the edit text is empty display a toast message.
    Toast.makeText(MainActivity.this, "Model not loaded yet, please wait...", Toast.LENGTH_SHORT).show();
    return;
}
```

### Prompt Template
On its own, this model's answers can be very long. To format the AI assistant's answers, you can adjust the prompt template. 
```java
String promptQuestion = userMsgEdt.getText().toString();
String promptQuestion_formatted = "<system>You are a helpful AI assistant. Answer in two paragraphs or less<|end|><|user|>"+promptQuestion+"<|end|>\n<assistant|>";
Log.i("GenAI: prompt question", promptQuestion_formatted);
```
You can also include limits such as a max_length or length_penalty to your liking. 
```java
generatorParams.setSearchOption("length_penalty", 1000);
generatorParams.setSearchOption("max_length", 500);
```
NOTE: Including a max_length will cut off the assistant's answer once reaching the maximum number of tokens rather than formatting a complete response.

## Run the App
### Build the generate() API
NOTE: **SKIP THIS STEP!** This project includes an .aar package, so does not need to be built. 

Follow the instructions on how to [Build the generate() API from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html) for the [java API](https://onnxruntime.ai/docs/genai/api/java.html).

### Download Android Studio
You will be using [Android Studio](https://developer.android.com/studio) to run the app.

### Download the App
Clone the [ONNX Runtime Inference Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/c29d8edd6d010a2649d69f84f54539f1062d776d) repository.

### Enable Developer Mode on Mobile
On your Android Mobile device, go to "Settings > About Phone > Software information" and tap the "Build Number" tile repeatedly until you see the message “You are now in developer mode”. In "Developer Options", turn on Wireless or USB debugging.

### Open Project in Android Studio
Open the Phi-3 mobile app in Android Studio (onnxruntime-inference-examples/mobile/examples/phi-3/android/app).

### Connect Device
To run the app on a device, follow the instructions from the Running Devices tab on the right side panel. You can connect through Wi-Fi or USB.
![WiFi Instructions](../../../images/phi3_MobileTutorial_RunDevice.png)
#### Pair over Wi-Fi
![WiFi Instructions](../../../images/phi3_MobileTutorial_WiFi.png)

### Manage Devices
You can manage/change devices and device model through the Device Manager tab on the right side panel.
![WiFi Instructions](../../../images/phi3_MobileTutorial_DeviceManager.png)

### Downloading the App
Once your device is connected, run the app by using the play button on the top panel. Downloading all packages will take ~10-15 minutes. If you submit a prompt before downloads are complete, you will encounter an error message. Once completed, the logcat (the cat tab on the bottom left panel) will display an "All downloads complete" message.

![WiFi Instructions](../../../images/phi3_MobileTutorial_Error.png)

### Ask questions
Now that the app is downloaded, you can start asking questions!
![Example Prompt 1](../../../images/phi3_MobileTutorial_ex1.png)
![Example Prompt 2](../../../images/phi3_MobileTutorial_ex2.png)
![Example Prompt 3](../../../images/phi3_MobileTutorial_ex3.png)
