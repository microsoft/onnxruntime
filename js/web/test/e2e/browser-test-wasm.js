it('Browser E2E testing - WebAssembly backend', async function () {
  await testFunction(ort, { executionProviders: ['wasm'] });
});
