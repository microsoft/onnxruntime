<script setup>
import { ref } from 'vue'

defineProps({
  msg: String,
})

const ortState = ref(0)
const ortLog = ref('Ready.')

const setOrtState = (state) => {
  ortState.value = state;
};

const setOrtLog = (log) => {
  ortLog.value = log;
};

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


</script>

<template>
    <div>
      <label>
        <input type="checkbox" title="Multi-thread" id="cb-mt" />
        Multi-thread
      </label>
      <label>
        <input type="checkbox" title="Proxy" id="cb-px" />
        Proxy
      </label>
      <button id="btn-load" @click="loadModel" :disabled="ortState !== 0">
        Load Model
      </button>
      <button id="btn-run" @click="runTest" :disabled="ortState !== 2">
        Run Test
      </button>
      <div id="ortstate">{{ortState}}</div>
      <div id="ortlog">{{ortLog}}</div>
    </div>
</template>

<style scoped>
.read-the-docs {
  color: #888;
}
</style>
