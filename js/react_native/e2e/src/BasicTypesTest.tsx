// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as React from 'react';
import { ActivityIndicator, Button, ScrollView, StyleSheet, Text, View, Platform } from 'react-native';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Buffer } from 'buffer';
import RNFS from 'react-native-fs';

interface TestModel {
  name: string;
  asset: string;
  dataType: keyof Tensor.DataTypeMap;
  description: string;
}

interface TestResult {
  model: string;
  status: 'pending' | 'running' | 'success' | 'error';
  message?: string;
  duration?: number;
}

interface State {
  testResults: TestResult[];
  isRunning: boolean;
}

const TEST_MODELS: TestModel[] = [
  {
    name: 'Bool',
    asset: 'test_types_bool.onnx',
    dataType: 'bool',
    description: 'Test boolean data type',
  },
  {
    name: 'Float',
    asset: 'test_types_float.ort',
    dataType: 'float32',
    description: 'Test float32 data type',
  },
  {
    name: 'Double',
    asset: 'test_types_double.onnx',
    dataType: 'float64',
    description: 'Test float64 data type',
  },
  {
    name: 'Int8',
    asset: 'test_types_int8.ort',
    dataType: 'int8',
    description: 'Test int8 data type',
  },
  {
    name: 'UInt8',
    asset: 'test_types_uint8.ort',
    dataType: 'uint8',
    description: 'Test uint8 data type',
  },
  {
    name: 'Int32',
    asset: 'test_types_int32.ort',
    dataType: 'int32',
    description: 'Test int32 data type',
  },
  {
    name: 'Int64',
    asset: 'test_types_int64.ort',
    dataType: 'int64',
    description: 'Test int64 data type',
  },
];

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 20,
    color: '#666',
  },
  buttonContainer: {
    marginBottom: 20,
  },
  resultsContainer: {
    flex: 1,
  },
  testItem: {
    backgroundColor: '#fff',
    padding: 15,
    marginBottom: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  testHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  testName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
  },
  testDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
  testFile: {
    fontSize: 12,
    color: '#999',
    marginBottom: 5,
  },
  statusSuccess: {
    fontSize: 24,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  statusError: {
    fontSize: 24,
    color: '#F44336',
    fontWeight: 'bold',
  },
  statusPending: {
    fontSize: 24,
    color: '#999',
  },
  successMessage: {
    fontSize: 12,
    color: '#4CAF50',
    marginTop: 5,
  },
  errorMessage: {
    fontSize: 12,
    color: '#F44336',
    marginTop: 5,
  },
});

const readAsset = async (asset: string): Promise<Buffer> => {
  if (Platform.OS === 'android') {
    return Buffer.from(await RNFS.readFileAssets(asset, 'base64'), 'base64');
  } else {
    return Buffer.from(await RNFS.readFile(`${RNFS.MainBundlePath}/${asset}`, 'base64'), 'base64');
  }
};

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export default class BasicTypesTest extends React.PureComponent<{}, State> {
  private sessions: Map<string, InferenceSession> = new Map();

  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  constructor(props: {} | Readonly<{}>) {
    super(props);

    this.state = {
      testResults: TEST_MODELS.map(model => ({
        model: model.name,
        status: 'pending',
      })),
      isRunning: false,
    };
  }

  async componentWillUnmount(): Promise<void> {
    // Dispose all sessions when leaving the page
    console.log('Disposing all basic type test sessions');
    for (const [name, session] of this.sessions.entries()) {
      try {
        await session.release();
        console.log(`Disposed session: ${name}`);
      } catch (err) {
        console.error(`Error disposing session ${name}:`, err);
      }
    }
    this.sessions.clear();
  }

  updateTestResult = (index: number, update: Partial<TestResult>) => {
    this.setState(prevState => {
      const newResults = [...prevState.testResults];
      newResults[index] = { ...newResults[index], ...update };
      return { testResults: newResults };
    });
  };

  createTensorData = (dataType: keyof Tensor.DataTypeMap, size = 5): Tensor.DataType => {
    switch (dataType) {
      case 'bool': {
        const data = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i % 2;
        }
        return data;
      }
      case 'float32': {
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i * 1.5;
        }
        return data;
      }
      case 'float64': {
        const data = new Float64Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i * 2.5;
        }
        return data;
      }
      case 'int8': {
        const data = new Int8Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i - 2;
        }
        return data;
      }
      case 'uint8': {
        const data = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i;
        }
        return data;
      }
      case 'int32': {
        const data = new Int32Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = i * 10;
        }
        return data;
      }
      case 'int64': {
        const data = new BigInt64Array(size);
        for (let i = 0; i < size; i++) {
          data[i] = BigInt(i);
        }
        return data;
      }
      default:
        return new Float32Array(size);
    }
  };

  runSingleTest = async (model: TestModel, index: number): Promise<void> => {
    const startTime = Date.now();
    this.updateTestResult(index, { status: 'running' });

    try {
      // Get model path
      const bytes = await readAsset(model.asset);

      // Create session
      const session = await InferenceSession.create(bytes);
      this.sessions.set(model.name, session);

      // Create input tensor
      const inputData = this.createTensorData(model.dataType, 5);
      const inputTensor = new Tensor(model.dataType, inputData, [1, 5]);

      // Run inference
      const feeds: Record<string, Tensor> = {};
      feeds[session.inputNames[0]] = inputTensor;
      const output = await session.run(feeds);

      // Verify output
      const outputTensor = output[session.outputNames[0]];
      if (outputTensor && outputTensor.data) {
        const duration = Date.now() - startTime;
        this.updateTestResult(index, {
          status: 'success',
          message: `Output shape: [${outputTensor.dims.join(', ')}], Duration: ${duration}ms`,
          duration,
        });
        console.log(`${model.name} test passed in ${duration}ms`);
      } else {
        throw new Error('No output received');
      }
    } catch (err) {
      const duration = Date.now() - startTime;
      const errorMessage = err instanceof Error ? err.message : String(err);
      this.updateTestResult(index, {
        status: 'error',
        message: errorMessage,
        duration,
      });
      console.error(`${model.name} test failed:`, errorMessage);
    }
  };

  runAllTests = async (): Promise<void> => {
    this.setState({ isRunning: true });

    // Reset all results
    this.setState({
      testResults: TEST_MODELS.map(model => ({
        model: model.name,
        status: 'pending',
      })),
    });

    // Clear existing sessions
    for (const [name, session] of this.sessions.entries()) {
      try {
        await session.release();
      } catch (err) {
        console.error(`Error disposing session ${name}:`, err);
      }
    }
    this.sessions.clear();

    // Run tests sequentially
    for (let i = 0; i < TEST_MODELS.length; i++) {
      await this.runSingleTest(TEST_MODELS[i], i);
    }

    this.setState({ isRunning: false });
  };

  render(): React.JSX.Element {
    const { testResults, isRunning } = this.state;

    return (
      <View style={styles.container}>
        <Text style={styles.title}>Basic Types Test</Text>
        <Text style={styles.subtitle}>
          Test ONNX Runtime with various data types
        </Text>

        <View style={styles.buttonContainer}>
          <Button
            title={isRunning ? 'Running Tests...' : 'Run All Tests'}
            onPress={this.runAllTests}
            disabled={isRunning}
            accessibilityLabel="run-tests-button"
          />
        </View>

        <ScrollView style={styles.resultsContainer}>
          {TEST_MODELS.map((model, index) => {
            const result = testResults[index];
            return (
              <View key={model.name} style={styles.testItem}>
                <View style={styles.testHeader}>
                  <Text style={styles.testName}>{model.name}</Text>
                  {result.status === 'running' && (
                    <ActivityIndicator size="small" color="#007AFF" accessibilityLabel="statusRunning" />
                  )}
                  {result.status === 'success' && (
                    <Text style={styles.statusSuccess} accessibilityLabel="statusSuccess">✓</Text>
                  )}
                  {result.status === 'error' && (
                    <Text style={styles.statusError} accessibilityLabel="statusError">✗</Text>
                  )}
                  {result.status === 'pending' && (
                    <Text style={styles.statusPending} accessibilityLabel="statusPending">○</Text>
                  )}
                </View>
                <Text style={styles.testDescription}>{model.description}</Text>
                {result.message && (
                  <Text
                    style={
                      result.status === 'error'
                        ? styles.errorMessage
                        : styles.successMessage
                    }
                  >
                    {result.message}
                  </Text>
                )}
              </View>
            );
          })}
        </ScrollView>
      </View>
    );
  }
}
