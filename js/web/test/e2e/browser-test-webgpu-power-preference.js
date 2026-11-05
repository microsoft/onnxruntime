// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WebGPU power preference', async function () {
  const preference = __ort_arg_power_preference;
  ort.env.webgpu.powerPreference = preference === 'unset' ? undefined : preference;

  const options = {
    executionProviders: [__ort_arg_ep_form === 'object' ? { name: 'webgpu' } : 'webgpu'],
    extra: { session: { disable_cpu_ep_fallback: '1' } },
  };
  let expectedPreference = ort.env.webgpu.powerPreference ?? 'high-performance';
  if (typeof __ort_arg_override !== 'undefined') {
    expectedPreference = __ort_arg_override;
    if (typeof __ort_arg_nested_extra !== 'undefined') {
      options.extra.ep = { webgpuexecutionprovider: { powerPreference: expectedPreference } };
    } else {
      options.extra['ep.webgpuexecutionprovider.powerPreference'] = expectedPreference;
    }
  }
  if (typeof __ort_arg_ignored_adapter_settings !== 'undefined') {
    ort.env.webgpu.adapter = {};
    ort.env.webgpu.forceFallbackAdapter = true;
  }

  const requestAdapter = navigator.gpu.requestAdapter;
  const requests = [];
  navigator.gpu.requestAdapter = function (requestOptions) {
    requests.push(requestOptions);
    return requestAdapter.call(this, requestOptions);
  };
  try {
    if (expectedPreference === 'invalid') {
      let error;
      try {
        await testFunction(ort, options);
      } catch (caught) {
        error = caught;
      }
      assert(error instanceof Error);
      assert(/Invalid powerPreference setting|Invalid power preference/.test(error.message));
      assert(requests.length === 0);
    } else {
      await testFunction(ort, options);
      assert(requests.length === 1);
      assert(requests[0]?.powerPreference === expectedPreference);
    }
  } finally {
    navigator.gpu.requestAdapter = requestAdapter;
  }
});
