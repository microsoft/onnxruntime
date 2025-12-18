// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as React from 'react';
import { Button, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import MNISTTest from './MNISTTest';
import BasicTypesTest from './BasicTypesTest';

type Page = 'home' | 'mnist' | 'basic-types';

interface State {
  currentPage: Page;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
    color: '#333',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 18,
    marginBottom: 30,
    color: '#666',
    textAlign: 'center',
  },
  buttonContainer: {
    width: '100%',
    marginBottom: 30,
    alignItems: 'center',
  },
  buttonWrapper: {
    width: '80%',
    marginBottom: 10,
  },
  description: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  header: {
    padding: 10,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  testContent: {
    flex: 1,
  },
});

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export default class App extends React.PureComponent<{}, State> {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  constructor(props: {} | Readonly<{}>) {
    super(props);

    this.state = {
      currentPage: 'home',
    };
  }

  navigateTo = (page: Page) => {
    this.setState({ currentPage: page });
  };

  renderHome(): React.JSX.Element {
    return (
      <SafeAreaView style={styles.container}>
        <ScrollView contentContainerStyle={styles.scrollContent}>
          <Text style={styles.title}>ONNX Runtime E2E Tests</Text>
          <Text style={styles.subtitle}>Select a test to run:</Text>

          <View style={styles.buttonContainer}>
            <View style={styles.buttonWrapper}>
              <Button
                title="MNIST Test"
                onPress={() => this.navigateTo('mnist')}
                accessibilityLabel="mnist-test-button"
              />
            </View>
            <Text style={styles.description}>
              Test MNIST model with image classification
            </Text>
          </View>

          <View style={styles.buttonContainer}>
            <View style={styles.buttonWrapper}>
              <Button
                title="Basic Types Test"
                onPress={() => this.navigateTo('basic-types')}
                accessibilityLabel="basic-types-test-button"
              />
            </View>
            <Text style={styles.description}>
              Test various data types with basic models
            </Text>
          </View>
        </ScrollView>
      </SafeAreaView>
    );
  }

  renderTestPage(): React.JSX.Element {
    const { currentPage } = this.state;

    let testComponent: React.JSX.Element;
    switch (currentPage) {
      case 'mnist':
        testComponent = <MNISTTest />;
        break;
      case 'basic-types':
        testComponent = <BasicTypesTest />;
        break;
      default:
        testComponent = <Text>Unknown test</Text>;
    }

    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <Button
            title="← Back to Home"
            onPress={() => this.navigateTo('home')}
            accessibilityLabel="back-button"
          />
        </View>
        <ScrollView style={styles.testContent}>
          {testComponent}
        </ScrollView>
      </SafeAreaView>
    );
  }

  render(): React.JSX.Element {
    const { currentPage } = this.state;

    if (currentPage === 'home') {
      return this.renderHome();
    }

    return this.renderTestPage();
  }
}
