'use client';

import { useState } from 'react';

export default function OnnxTestBar() {
    const [ortState, setOrtState] = useState('');

    const loadModel = async () => {
        setOrtState('Loading model...');
        try {
            const { createTestSession } = await import('../app/onnx-helper');
            await createTestSession(document.getElementById('cb-mt').checked, document.getElementById('cb-px').checked);
        } catch (e) {
            setOrtState(`Failed to load model: ${e}`);
            return;
        }
        setOrtState('Model loaded.');
    };

    const runTest = async () => {
        setOrtState('Running model test...');
        const { runTestSessionAndValidate } = await import('../app/onnx-helper');
        const testResult = await runTestSessionAndValidate();
        setOrtState(`Test result: ${testResult}`);
    };

    return (
        <div>
            <label><input type="checkbox" title="Multi-thread" id="cb-mt" />Multi-thread</label>
            <label><input type="checkbox" title='Proxy' id="cb-px" />Proxy</label>
            <button onClick={loadModel}>Load Model</button>
            <button onClick={runTest}>Run Test</button>
            <div>{ortState}</div>
        </div>
    );
}