it('Browser E2E testing - WebAssembly backend (no threads)', async function () {
  ort.env.wasm.numThreads = 1;
  await testFunction(ort, { executionProviders: ['wasm'] });
});
