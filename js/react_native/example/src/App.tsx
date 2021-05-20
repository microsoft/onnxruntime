// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import *as React from 'react';
import{Button, Image, Text, View} from 'react-native';
import{InferenceSession, Tensor} from 'onnxruntime-react-native';
import MNIST, {MNISTInput, MNISTOutput, MNISTResult, } from './mnist-data-handler';
import{Buffer} from 'buffer';

interface Duration {
preprocess:
  number;
inference:
  number;
postprocess:
  number;
}

interface State {
session:
  InferenceSession | null;
output:
  string | null;
duration:
  Duration | null;
imagePath:
  string | null;
}

// eslint-disable-next-line @typescript-eslint/ban-types
export default class App extends React.PureComponent<{}, State> {
  // eslint-disable-next-line @typescript-eslint/ban-types
  constructor(props : {} | Readonly<{}>) {
    super(props);

    this.state = {
      session : null,
      output : null,
      duration : null,
      imagePath : null,
    };
  }

  // Load a model when an app is loading
  async componentDidMount() : Promise<void> {
    if (!this.state.session) {
      try {
        const imagePath = await MNIST.getImagePath();
        this.setState({imagePath});

        const modelPath = await MNIST.getLocalModelPath();
        const session : InferenceSession = await InferenceSession.create(modelPath);
        this.setState({session});
      } catch (err) {
        console.log(err.message);
      }
    }
  }

  // Run a model with a given image
  infer = async() : Promise<void> => {
    try {
      let preprocessTime = 0;
      let inferenceTime = 0;
      let postprocessTime = 0;

      const options : InferenceSession.RunOptions = {};

      let startTime = Date.now();
      const mnistInput : MNISTInput = await MNIST.preprocess(this.state.imagePath !);
      const input : {[name:string] : Tensor} = {};
      for (const key in mnistInput) {
        if (Object.hasOwnProperty.call(mnistInput, key)) {
          const buffer = Buffer.from(mnistInput[key].data, 'base64');
          const tensorData =
              new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / Float32Array.BYTES_PER_ELEMENT);
          input[key] = new Tensor(mnistInput[key].type as keyof Tensor.DataTypeMap, tensorData, mnistInput[key].dims);
        }
      }
      preprocessTime = Date.now() - startTime;

      startTime = Date.now();
      const output : InferenceSession.ReturnType =
          await this.state.session !.run(input, this.state.session !.outputNames, options);
      inferenceTime = Date.now() - startTime;

      startTime = Date.now();
      const mnistOutput : MNISTOutput = {};
      for (const key in output) {
        if (Object.hasOwnProperty.call(output, key)) {
          const buffer = (output[key].data as Float32Array).buffer;
          const tensorData = {
            data : Buffer.from(buffer, 0, buffer.byteLength).toString('base64'),
          };
          mnistOutput[key] = tensorData;
        }
      }
      const result : MNISTResult = await MNIST.postprocess(mnistOutput);
      postprocessTime = Date.now() - startTime;

      this.setState({
        output : result.result,
        duration : {
          preprocess : preprocessTime,
          inference : inferenceTime,
          postprocess : postprocessTime,
        },
      });
    } catch (err) {
      console.log(err.message);
    }
  };

  render() : JSX.Element {
    const {output, duration, imagePath} = this.state;

    return (
      <View>
        <Text>{'\n'}</Text>
        <Button
          title={'Run'}
          disabled={!this.state.session}
          onPress={this.infer}
        />
        {imagePath && (
          <Image
            source={
      {
      uri:
        imagePath
      }}
            style={{
              height: 200,
              width: 200,
              resizeMode: 'stretch',
            }}
          />
        )}
        {output && (
          <Text>
            Result: {output} {'\n'}
            {'--------------------------------\n'}
            Preprocess time taken: {duration?.preprocess} ms {'\n'}
            Inference time taken: {duration?.inference} ms {'\n'}
            Postprocess time taken: {duration?.postprocess} ms {'\n'}
          </Text>
        )}
      </View>
    );
  }
}
