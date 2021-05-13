it('Browser E2E testing - WebGL backend', async function () {
  await testFunction(ort, { executionProviders: ['webgl'] });
});
