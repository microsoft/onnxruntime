---
title: Custom Excel Functions for BERT Tasks in JavaScript
description:  Custom Excel Functions for BERT Tasks in JavaScript
parent: Web
grand_parent: Tutorials
has_children: false
nav_order: 8
---

# ONNX Runtime Custom Excel Functions for BERT NLP Tasks in JavaScript
{: .no_toc }

In this tutorial we will look at how we can create custom Excel functions (`ORT.Sentiment()` and `ORT.Question()`) to implement BERT NLP models with ONNX Runtime Web to enable deep learning in spreadsheet tasks. The inference happens locally, right in Excel! 

<img src="../../../images/bert-excel.gif" width="560" height="315" alt="Image of browser inferencing on sample images."/>


<!-- <div class="embed-container">
  <iframe
      src="https://www.youtube.com/embed/wuSxWGY_Sjg"
      width="560" height="315"
      frameborder="0"
      allowfullscreen="true">
  </iframe>
</div> -->

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Prerequisites
- [Node.js](https://nodejs.org/en/)
- Office connected to a Microsoft 365 subscription (including Office on the web). If you don't already have Office, you can join the [Microsoft 365 developer program to get a free](https://developer.microsoft.com/office/dev-program), 90-day renewable Microsoft 365 subscription to use during development.
- [See Office Add-ins tutorial for more information](https://learn.microsoft.com/office/dev/add-ins/tutorials/excel-tutorial-create-custom-functions?source=recommendations&tabs=excel-windows#prerequisites)

## What are Custom Functions?

Excel has many native functions like `SUM()` that you are likely familiar with. Custom functions are a useful tool to create and add new functions to Excel by defining those functions in JavaScript as part of an add-in. These functions can be accessed within Excel just as you would any native function in Excel.

## Creating the Custom Function project

Now that we know what custom functions are lets look at how we can create functions that will inference a model locally to get the sentiment text in a cell or extract information from a cell by asking a question and the answer being returned to the cell.

- If you plan to follow along, [clone the project that we will discuss in this blog](https://github.com/cassiebreviu/bert-excel-addin-ort-web). This project was created with the template project from the Yeoman CLI. [Learn more in this quickstart about the base projects](https://learn.microsoft.com/office/dev/add-ins/tutorials/excel-tutorial-create-custom-functions).

- Run the following commands to install the packages and build the project.

```bash
npm install
npm run build
```

- The below commend will run the add-in in Excel web and side load the add-in to the spreadsheet that is provided in the command.

```bash
// Command to run on the web.
// Replace "{url}" with the URL of an Excel document.
npm run start:web -- --document {url}
```

- Use the following command to run in the Excel client.

```bash
// Command to run on desktop (Windows or Mac)
npm run start:desktop
```

- The first time you run the project there will be two prompts:
  - One will ask to `Enable Developer Mode`. This is required for sideloading plugins.
  - Next when prompted accept the certificate for the plugin service.

- To access the custom function type `=ORT.Sentiment("TEXT")` and `=ORT.Question("QUESTION","CONTEXT")` in an empty cell and pass in the parameters.

Now we are ready to jump into the code!

## The `manifest.xml` file

 The `manifest.xml` file specifies that all custom functions belong to the `ORT` namespace. You'll use the namespace to access the custom functions in Excel. Update the values in the `mainfest.xml` to `ORT`.

```xml
<bt:String id="Functions.Namespace" DefaultValue="ORT"/>
<ProviderName>ORT</ProviderName>
```

Learn more about the configuration of the [mainfest file here](https://learn.microsoft.com/office/dev/add-ins/develop/configure-your-add-in-to-use-a-shared-runtime#configure-the-manifest).

## The `functions.ts` file

In the [`function.ts`](https://github.com/cassiebreviu/bert-excel-addin-ort-web/blob/main/src/functions/functions.ts) file we define the functions name, parameters, logic and return type.

- Import the functions `inferenceQuestion` and `inferenceSentiment` at the top of the `function.ts` file. (We will go over the logic in these functions later in this tutorial.)

```javascript
/* global console */
import { inferenceQuestion } from "./bert/inferenceQuestion";
import { inferenceSentiment } from "./bert/inferenceSentiment";
```
- Next add the `sentiment` and `question` functions.

```javascript
/**
* Returns the sentiment of a string.
* @customfunction
* @param text Text string
* @returns sentiment string.
*/
export async function sentiment(text: string): Promise<string> {
const result = await inferenceSentiment(text);
console.log(result[1][0]);
return result[1][0].toString();
}
/**
 * Returns the sentiment of a string.
 * @customfunction
 * @param question Question string
 * @param context Context string
 * @returns answer string.
 */
export async function question(question: string, context: string): Promise<string> {
const result = await inferenceQuestion(question, context);
if (result.length > 0) {
    console.log(result[0].text);
    return result[0].text.toString();
}
return "Unable to find answer";
}
```

## The `inferenceQuestion.ts` file

The [`inferenceQuestion.ts`](https://github.com/cassiebreviu/bert-excel-addin-ort-web/blob/main/src/functions/bert/inferenceQuestion.ts) file has the logic to process the Question and Answer BERT Model. This model was created using [this tutorial](https://onnxruntime.ai/docs/tutorials/azureml.html#obtain-and-convert-pytorch-model-to-onnx-format). Then we used ORT Quantization tool to reduce the size of the model. Learn more about [quantization here](https://onnxruntime.ai/docs/performance/quantization.html).


- First import `onnxruntime-web` and the helper functions from `question_answer.ts`. The `question_answer.ts` is an edited version from the tensorflow example found [here](https://github.com/tensorflow/tfjs-models/blob/master/qna/src/question_and_answer.ts). You can find the edited version in the source for this project [here](https://github.com/cassiebreviu/bert-excel-addin-ort-web/blob/main/src/functions/bert/question_answer.ts).

```javascript
/* eslint-disable no-undef */
import * as ort from "onnxruntime-web";
import { create_model_input, Feature, getBestAnswers, Answer } from "./utils/question_answer";
```

- The `inferenceQuestion` function will take in the question and context and provide the answers based on the inference result. Then we set the path to the model. This path is set in the `webpack.config.js` with the `CopyWebpackPlugin`. This plugin copies the assets needed on build to the `dist` folder.

```javascript
export async function inferenceQuestion(question: string, context: string): Promise<Answer[]> {
  const model: string = "./bert-large-uncased-int8.onnx";
```
- Now lets create the ONNX Runtime Inference Session and set the options. Learn more about all the `SessionOptions` [here](https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html).

```javascript
  // create session, set options
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    // executionProviders: ['webgl']
    graphOptimizationLevel: "all",
  };
  console.log("Creating session");
  const session = await ort.InferenceSession.create(model, options);
```
- Next we encode the `question` and `context` using the `create_model_input` function from the `question_answer.ts`. This returns the `Feature`.

```javascript
  // Get encoded ids from text tokenizer.
  const encoded: Feature = await create_model_input(question, context);
  console.log("encoded", encoded);
```
```javascript
  export interface Feature {
    input_ids: Array<any>;
    input_mask: Array<any>;
    segment_ids: Array<any>;
    origTokens: Token[];
    tokenToOrigMap: { [key: number]: number };
}
```
- Now that we have the `encoded` `Feature`, we need to create arrays (`input_ids`, `attention_mask`, and `token_type_ids`) of type `BigInt` to create `ort.Tensor` input.

```javascript
  // Create arrays of correct length
  const length = encoded.input_ids.length;
  var input_ids = new Array(length);
  var attention_mask = new Array(length);
  var token_type_ids = new Array(length);

  // Get encoded.input_ids as BigInt
  input_ids[0] = BigInt(101);
  attention_mask[0] = BigInt(1);
  token_type_ids[0] = BigInt(0);
  var i = 0;
  for (; i < length; i++) {
    input_ids[i + 1] = BigInt(encoded.input_ids[i]);
    attention_mask[i + 1] = BigInt(1);
    token_type_ids[i + 1] = BigInt(0);
  }
  input_ids[i + 1] = BigInt(102);
  attention_mask[i + 1] = BigInt(1);
  token_type_ids[i + 1] = BigInt(0);

  console.log("arrays", input_ids, attention_mask, token_type_ids);
```
- Create `ort.Tensor` from the `Arrays`.

```javascript
  const sequence_length = input_ids.length;
  var input_ids_tensor: ort.Tensor = new ort.Tensor("int64", BigInt64Array.from(input_ids), [1, sequence_length]);
  var attention_mask_tensor: ort.Tensor = new ort.Tensor("int64", BigInt64Array.from(attention_mask), [ 1, sequence_length]);
  var token_type_ids_tensor: ort.Tensor = new ort.Tensor("int64", BigInt64Array.from(token_type_ids), [ 1, sequence_length]);
```
 - We are ready to run inference! Here we create the `OnnxValueMapType` (input object) and `FetchesType` (return labels). You can send in the object and string array without declaring the type however adding the types are useful.

```javascript
  const model_input: ort.InferenceSession.OnnxValueMapType = {
    input_ids: input_ids_tensor,
    input_mask: attention_mask_tensor,
    segment_ids: token_type_ids_tensor,
  };
  const output_names: ort.InferenceSession.FetchesType = ["start_logits", "end_logits"];
  const output = await session.run(model_input, output_names);
  const result_length = output["start_logits"].data.length;
```
- Next loop through the result and create a `number` array from the resulting `start_logits` and `end_logits`.

```javascript
  const start_logits: number[] = Array(); 
  const end_logits: number[] = Array(); 
  console.log("start_logits", start_logits);
  console.log("end_logits", end_logits);
  for (let i = 0; i <= result_length; i++) {
    start_logits.push(Number(output["start_logits"].data[i]));
  }
  for (let i = 0; i  <= result_length; i++) {
    end_logits.push(Number(output["end_logits"].data[i]));
  }
```

- Lastly we will call [`getBestAnswers`](https://github.com/cassiebreviu/bert-excel-addin-ort-web/blob/main/src/functions/bert/question_answer.ts#L142) from `question_answer.ts`. This will take result and do the post processing to get the answer from the inference result. 

```javascript
  const answers: Answer[] = getBestAnswers(
    start_logits,
    end_logits,
    encoded.origTokens,
    encoded.tokenToOrigMap,
    context
  );
  console.log("answers", answers);
  return answers;
}
```

- The `answers` are then returned back to the `functions.ts` `question`, the resulting string is returned and populated into the Excel cell.

```javascript
export async function question(question: string, context: string): Promise<string> {
  const result = await inferenceQuestion(question, context);
  if (result.length > 0) {
    console.log(result[0].text);
    return result[0].text.toString();
  }
  return "Unable to find answer";
}
```
- Now you can run the below command to build and side load the add-in to your Excel spreadsheet!

```bash
// Command to run on the web.
// Replace "{url}" with the URL of an Excel document.
npm run start:web -- --document {url}
```

That is a breakdown for the `ORT.Question()` custom function, next we will breakdown how the `ORT.Sentiment()` is implemented.

## The `inferenceSentiment.ts` file

The [`inferenceSentiment.ts`](https://github.com/cassiebreviu/bert-excel-addin-ort-web/blob/main/src/functions/bert/inferenceSentiment.ts) is the logic to inference and get sentiment for text in an Excel cell. The code here is augmented from [this example](https://github.com/jobergum/browser-ml-inference). Let's jump in and learn how this part works.

- First lets import the packages needed. As you will see in this tutorial the `bertProcessing` function will create our model input.  `bert_tokenizer` is the JavaScript tokenizer for BERT models. `onnxruntime-web` enables inference in JavaScript on the browser.

```javascript
/* eslint-disable no-undef */
import * as bertProcessing from "./bertProcessing";
import * as ort from "onnxruntime-web";
import { EMOJIS } from "./emoji";
import { loadTokenizer } from "./bert_tokenizer";
```
- Now lets load the quantized BERT model that has been finetuned for sentiment analysis. Then create the `ort.InferenceSession` and `ort.InferenceSession.SessionOptions`.

```javascript
export async function inferenceSentiment(text: string) {
  // Set model path.
  const model: string = "./xtremedistill-go-emotion-int8.onnx";
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    // executionProviders: ['webgl']
    graphOptimizationLevel: "all",
  };
  console.log("Creating session");
  const session = await ort.InferenceSession.create(model, options);
```
- Next we tokenize the text to create the `model_input` and send it to `session.run` with the output label `output_0` to get the inference result.

```javascript
  // Get encoded ids from text tokenizer.
  const tokenizer = loadTokenizer();
  const encoded = await tokenizer.then((t) => {
    return t.tokenize(text);
  });
  console.log("encoded", encoded);
  const model_input = await bertProcessing.create_model_input(encoded);
  console.log("run session");
  const output = await session.run(model_input, ["output_0"]);
  const outputResult = output["output_0"].data;
  console.log("outputResult", outputResult);
```
- Next we parse the output to get the top result and map it to the label, score and emoji.

```javascript
  let probs = [];
  for (let i = 0; i < outputResult.length; i++) {
    let sig = bertProcessing.sigmoid(outputResult[i]);
    probs.push(Math.floor(sig * 100));
  }
  console.log("probs", probs);
  const result = [];
  for (var i = 0; i < EMOJIS.length; i++) {
    const t = [EMOJIS[i], probs[i]];
    result[i] = t;
  }
  result.sort(bertProcessing.sortResult);
  console.log(result);
  const result_list = [];
  result_list[0] = ["Emotion", "Score"];
  for (i = 0; i < 6; i++) {
    result_list[i + 1] = result[i];
  }
  console.log(result_list);
  return result_list;
}

```
- The `result_list` is returned and parsed to return the top result to the Excel cell.

```javascript
export async function sentiment(text: string): Promise<string> {
  const result = await inferenceSentiment(text);
  console.log(result[1][0]);
  return result[1][0].toString();
}
```

- Now you can run the below command to build and side load the add-in to your Excel spreadsheet!

```bash
// Command to run on the web.
// Replace "{url}" with the URL of an Excel document.
npm run start:web -- --document {url}
```

## Conclusion

Here we went over the logic needed to create custom functions in an Excel add-in with JavaScript leveraging ONNX Runtime Web and open source models. From here you could take this logic and update to a specific model or use case you have. Be sure to check out the full source code which includes the tokenizers and pre/post processing to complete the above tasks.

## Additional resources
* [Publish Add-ins in VS Code](https://learn.microsoft.com/en-us/office/dev/add-ins/publish/publish-add-in-vs-code#using-visual-studio-code-to-publish)
* [Full source code for this example](https://github.com/cassiebreviu/bert-excel-addin-ort-web)
* [Office Add-ins documentation](https://docs.microsoft.com/office/dev/add-ins/overview/office-add-ins)
* [Excel Custom function Quickstart](https://learn.microsoft.com//office/dev/add-ins/quickstarts/excel-custom-functions-quickstart?tabs=excel-online)
