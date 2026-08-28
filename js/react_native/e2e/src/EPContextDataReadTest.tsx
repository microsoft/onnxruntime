// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as React from 'react';
import { ActivityIndicator, Button, ScrollView, StyleSheet, Text, View, Platform } from 'react-native';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Buffer } from 'buffer';
import RNFS from 'react-native-fs';

interface TestResult {
  name: string;
  status: 'pending' | 'running' | 'success' | 'error';
  message?: string;
}

interface State {
  testResults: TestResult[];
  isRunning: boolean;
}

// A plain (non-EPContext) model. Registering the read callback must not disturb a session that
// never needs external EPContext data.
const MODEL_ASSET = 'test_types_float.ort';

const CHECK_NAMES = [
  'Missing callback is rejected',
  'Non-function callback is rejected',
  'Missing maxDataSize is rejected',
  'Zero maxDataSize is rejected',
  'Negative maxDataSize is rejected',
  'Fractional maxDataSize is rejected',
  'Infinite maxDataSize is rejected',
  'Unsafe integer maxDataSize is rejected',
  'Valid option loads and runs a session',
  'Repeated create/release keeps the callback alive',
  'Failed load releases the callback state',
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
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    flexShrink: 1,
    paddingRight: 10,
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

// Avoids depending on TextEncoder, which is not available on every JS engine used by React Native.
const validCallback = (name: string): Uint8Array => {
  const bytes = new Uint8Array(name.length);
  for (let i = 0; i < name.length; i++) {
    bytes[i] = name.charCodeAt(i) % 256;
  }
  return bytes;
};

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export default class EPContextDataReadTest extends React.PureComponent<{}, State> {
  private session: InferenceSession | undefined;

  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  constructor(props: {} | Readonly<{}>) {
    super(props);

    this.state = {
      testResults: CHECK_NAMES.map((name) => ({ name, status: 'pending' })),
      isRunning: false,
    };
  }

  async componentWillUnmount(): Promise<void> {
    await this.releaseSession();
  }

  releaseSession = async (): Promise<void> => {
    if (this.session) {
      const session = this.session;
      this.session = undefined;
      try {
        await session.release();
      } catch (err) {
        console.error('Error releasing EPContext data read session:', err);
      }
    }
  };

  updateTestResult = (index: number, update: Partial<TestResult>) => {
    this.setState((prevState) => {
      const newResults = [...prevState.testResults];
      newResults[index] = { ...newResults[index], ...update };
      return { testResults: newResults };
    });
  };

  // Asserts that creating a session with the given options fails during option validation.
  expectRejected = async (bytes: Buffer, index: number, options: unknown): Promise<void> => {
    this.updateTestResult(index, { status: 'running' });
    let session: InferenceSession | undefined;
    try {
      session = await InferenceSession.create(bytes, options as InferenceSession.SessionOptions);
    } catch (err) {
      this.updateTestResult(index, {
        status: 'success',
        message: err instanceof Error ? err.message : String(err),
      });
      return;
    } finally {
      if (session) {
        await session.release();
      }
    }
    this.updateTestResult(index, {
      status: 'error',
      message: 'Session creation unexpectedly succeeded',
    });
  };

  runValidOptionCheck = async (bytes: Buffer, index: number): Promise<void> => {
    this.updateTestResult(index, { status: 'running' });
    try {
      let callbackCalls = 0;
      this.session = await InferenceSession.create(bytes, {
        epContextDataRead: {
          callback: (name: string) => {
            callbackCalls++;
            return validCallback(name);
          },
          maxDataSize: 1024 * 1024,
        },
      });

      const feeds: Record<string, Tensor> = {};
      feeds[this.session.inputNames[0]] = new Tensor('float32', new Float32Array([0, 1, 2, 3, 4]), [1, 5]);
      const output = await this.session.run(feeds);
      const outputTensor = output[this.session.outputNames[0]];
      if (!outputTensor || !outputTensor.data) {
        throw new Error('No output received');
      }

      await this.releaseSession();

      // The model has no external EPContext data, so ONNX Runtime must not invoke the callback.
      if (callbackCalls !== 0) {
        throw new Error(`Callback was invoked ${callbackCalls} time(s) for a model without EPContext data`);
      }

      this.updateTestResult(index, {
        status: 'success',
        message: `Output shape: [${outputTensor.dims.join(', ')}], callback invocations: ${callbackCalls}`,
      });
    } catch (err) {
      await this.releaseSession();
      this.updateTestResult(index, {
        status: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  runLifecycleCheck = async (bytes: Buffer, index: number): Promise<void> => {
    this.updateTestResult(index, { status: 'running' });
    try {
      for (let i = 0; i < 5; i++) {
        const session = await InferenceSession.create(bytes, {
          epContextDataRead: {
            callback: validCallback,
            maxDataSize: 16,
          },
        });
        // Releasing must drop the native callback state without disturbing later sessions.
        await session.release();
      }
      this.updateTestResult(index, {
        status: 'success',
        message: 'Created and released 5 sessions',
      });
    } catch (err) {
      this.updateTestResult(index, {
        status: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  // A session that fails to construct must release the callback state right away, and must leave a
  // subsequent load unaffected.
  runFailedLoadCheck = async (bytes: Buffer, index: number): Promise<void> => {
    this.updateTestResult(index, { status: 'running' });
    try {
      let callbackCalls = 0;
      const corrupted = Buffer.from(bytes);
      corrupted.fill(0, 0, Math.min(32, corrupted.length));

      let rejected = false;
      let badSession: InferenceSession | undefined;
      try {
        badSession = await InferenceSession.create(corrupted, {
          epContextDataRead: {
            callback: (name: string) => {
              callbackCalls++;
              return validCallback(name);
            },
            maxDataSize: 1024,
          },
        });
      } catch {
        rejected = true;
      }
      if (badSession) {
        await badSession.release();
      }
      if (!rejected) {
        throw new Error('Loading a corrupted model unexpectedly succeeded');
      }
      if (callbackCalls !== 0) {
        throw new Error(`Callback was invoked ${callbackCalls} time(s) for a model that failed to load`);
      }

      // The next session must still load and release normally.
      const session = await InferenceSession.create(bytes, {
        epContextDataRead: { callback: validCallback, maxDataSize: 1024 },
      });
      await session.release();

      this.updateTestResult(index, {
        status: 'success',
        message: 'Rejected load did not disturb the next session',
      });
    } catch (err) {
      this.updateTestResult(index, {
        status: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  };

  runAllTests = async (): Promise<void> => {
    this.setState({
      isRunning: true,
      testResults: CHECK_NAMES.map((name) => ({ name, status: 'pending' })),
    });

    try {
      const bytes = await readAsset(MODEL_ASSET);

      await this.expectRejected(bytes, 0, { epContextDataRead: { maxDataSize: 1024 } });
      await this.expectRejected(bytes, 1, { epContextDataRead: { callback: 'not-a-function', maxDataSize: 1024 } });
      await this.expectRejected(bytes, 2, { epContextDataRead: { callback: validCallback } });
      await this.expectRejected(bytes, 3, { epContextDataRead: { callback: validCallback, maxDataSize: 0 } });
      await this.expectRejected(bytes, 4, { epContextDataRead: { callback: validCallback, maxDataSize: -1 } });
      await this.expectRejected(bytes, 5, { epContextDataRead: { callback: validCallback, maxDataSize: 1.5 } });
      await this.expectRejected(bytes, 6, {
        epContextDataRead: { callback: validCallback, maxDataSize: Number.POSITIVE_INFINITY },
      });
      await this.expectRejected(bytes, 7, {
        epContextDataRead: { callback: validCallback, maxDataSize: Number.MAX_SAFE_INTEGER + 2 },
      });

      await this.runValidOptionCheck(bytes, 8);
      await this.runLifecycleCheck(bytes, 9);
      await this.runFailedLoadCheck(bytes, 10);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.setState((prevState) => ({
        testResults: prevState.testResults.map((result) =>
          result.status === 'pending' || result.status === 'running' ? { ...result, status: 'error', message } : result
        ),
      }));
    }

    this.setState({ isRunning: false });
  };

  render(): React.JSX.Element {
    const { testResults, isRunning } = this.state;

    return (
      <View style={styles.container}>
        <Text style={styles.title}>EPContext Data Read Test</Text>
        <Text style={styles.subtitle}>Validate the epContextDataRead session option and its native lifetime</Text>

        <View style={styles.buttonContainer}>
          <Button
            title={isRunning ? 'Running Tests...' : 'Run All Tests'}
            onPress={this.runAllTests}
            disabled={isRunning}
            accessibilityLabel="run-tests-button"
          />
        </View>

        <ScrollView style={styles.resultsContainer}>
          {testResults.map((result) => (
            <View key={result.name} style={styles.testItem}>
              <View style={styles.testHeader}>
                <Text style={styles.testName}>{result.name}</Text>
                {result.status === 'running' && (
                  <ActivityIndicator size="small" color="#007AFF" accessibilityLabel="statusRunning" />
                )}
                {result.status === 'success' && (
                  <Text style={styles.statusSuccess} accessibilityLabel="statusSuccess">
                    ✓
                  </Text>
                )}
                {result.status === 'error' && (
                  <Text style={styles.statusError} accessibilityLabel="statusError">
                    ✗
                  </Text>
                )}
                {result.status === 'pending' && (
                  <Text style={styles.statusPending} accessibilityLabel="statusPending">
                    ○
                  </Text>
                )}
              </View>
              {result.message && (
                <Text style={result.status === 'error' ? styles.errorMessage : styles.successMessage}>
                  {result.message}
                </Text>
              )}
            </View>
          ))}
        </ScrollView>
      </View>
    );
  }
}
