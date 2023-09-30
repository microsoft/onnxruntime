// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as React from 'react';
import { Image, Text, TextInput, View } from 'react-native';
// onnxruntime-react-native package is installed when bootstraping
// eslint-disable-next-line import/no-extraneous-dependencies
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import MNIST, { MNISTInput, MNISTOutput, MNISTResult, } from './mnist-data-handler';
import { Buffer } from 'buffer';
import { readFile } from 'react-native-fs';

interface State {
  session:
  InferenceSession | null;
  output:
  string | null;
  imagePath:
  string | null;
}

// eslint-disable-next-line @typescript-eslint/ban-types
export default class App extends React.PureComponent<{}, State> {
  // eslint-disable-next-line @typescript-eslint/ban-types
  constructor(props: {} | Readonly<{}>) {
    super(props);

    this.state = {
      session: null,
      output: null,
      imagePath: null,
    };
  }

  // Load a model when an app is loading
  async componentDidMount(): Promise<void> {
    if (!this.state.session) {
      try {
        const imagePath = await MNIST.getImagePath();
        this.setState({ imagePath });

        const modelPath = await MNIST.getLocalModelPath();

        // test creating session with path
        console.log('Creating with path');
        const pathSession: InferenceSession = await InferenceSession.create(modelPath);
        pathSession.release();

        // and with bytes
        console.log('Creating with bytes');
        const base64Str = await readFile(modelPath, 'base64');
        const bytes = Buffer.from(base64Str, 'base64');
        const session: InferenceSession = await InferenceSession.create(bytes);
        this.setState({ session });

        console.log('Test session created');
        void await this.infer();
      } catch (err) {
        console.log(err.message);
      }
    }
  }

  // Run a model with a given image
  infer = async (): Promise<void> => {
    try {
      const options: InferenceSession.RunOptions = {};

      const mnistInput: MNISTInput = await MNIST.preprocess(this.state.imagePath!);
      const input: { [name: string]: Tensor } = {};
      for (const key in mnistInput) {
        if (Object.hasOwnProperty.call(mnistInput, key)) {
          const buffer = Buffer.from(mnistInput[key].data, 'base64');
          const tensorData =
            new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / Float32Array.BYTES_PER_ELEMENT);
          input[key] = new Tensor(mnistInput[key].type as keyof Tensor.DataTypeMap, tensorData, mnistInput[key].dims);
        }
      }

      const output: InferenceSession.ReturnType =
        await this.state.session!.run(input, this.state.session!.outputNames, options);

      const mnistOutput: MNISTOutput = {};
      for (const key in output) {
        if (Object.hasOwnProperty.call(output, key)) {
          const buffer = (output[key].data as Float32Array).buffer;
          const tensorData = {
            data: Buffer.from(buffer, 0, buffer.byteLength).toString('base64'),
          };
          mnistOutput[key] = tensorData;
        }
      }
      const result: MNISTResult = await MNIST.postprocess(mnistOutput);

      this.setState({
        output: result.result
      });
    } catch (err) {
      console.log(err.message);
    }
  };

  render(): JSX.Element {
    const { output, imagePath } = this.state;

    return (
      <View>
        <Text>{'\n'}</Text>
        {imagePath && (
          <Image
            source={{
              uri: imagePath,
            }}
            style={{
              height: 200,
              width: 200,
              resizeMode: 'stretch',
            }}
          />
        )}
        {output && (
          <TextInput accessibilityLabel='output'>
            Result: {output}
          </TextInput>
        )}
      </View>
    );
  }
}
