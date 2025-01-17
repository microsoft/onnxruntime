'use client';

import { useState } from 'react';

export default function OnnxTestBar() {
  const [ortState, setOrtState] = useState(0);
  const [ortLog, setOrtLog] = useState('Ready.');

  const loadModel = async () => {
    setOrtState(1);
    setOrtLog('Loading model...');
    try {
      const { createTestSession } = await import('./onnx-helper');
      await createTestSession(document.getElementById('cb-mt').checked, document.getElementById('cb-px').checked);
      setOrtState(2);
      setOrtLog('Model loaded.');
    } catch (e) {
      setOrtState(3);
      setOrtLog(`Failed to load model: ${e}`);
      return;
    }
  };

  const runTest = async () => {
    setOrtState(4);
    setOrtLog('Running model test...');
    try {
      const { runTestSessionAndValidate } = await import('./onnx-helper');
      const testResult = await runTestSessionAndValidate();
      setOrtState(5);
      setOrtLog(`Test result: ${testResult}`);
    } catch (e) {
      setOrtState(6);
      setOrtLog(`Failed to load model: ${e}`);
      return;
    }
  };

  return (
    <div>
      <label>
        <input type="checkbox" title="Multi-thread" id="cb-mt" />
        Multi-thread
      </label>
      <label>
        <input type="checkbox" title="Proxy" id="cb-px" />
        Proxy
      </label>
      <button id="btn-load" onClick={loadModel} disabled={ortState !== 0}>
        Load Model
      </button>
      <button id="btn-run" onClick={runTest} disabled={ortState !== 2}>
        Run Test
      </button>
      <div id="ortstate">{ortState}</div>
      <div id="ortlog">{ortLog}</div>
    </div>
  );
}
