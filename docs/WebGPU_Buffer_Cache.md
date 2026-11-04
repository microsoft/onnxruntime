# WebGPU storage buffer bucket configuration

The native WebGPU EP uses a bucket cache for storage buffers by default. Each
allocation is rounded up to a bucket size, and each bucket retains a limited
number of free buffers. A growing dynamic KV cache can cross a large size step
or need more same-sized buffers than the default table retains.

Use `storageBufferBucketLimits` to replace the storage bucket table with sizes
and limits appropriate for the model. This minimal ONNX Runtime Web example
illustrates KV-sized buckets; a complete model table should also include smaller
sizes needed by intermediate tensors, so those allocations do not round up to
4 MiB:

```typescript
const session = await ort.InferenceSession.create(model, {
  executionProviders: [
    {
      name: "webgpu",
      storageBufferCacheMode: "bucket",
      storageBufferBucketLimits: {
        4194304: 24,
        5242880: 24,
        6291456: 24,
        8388608: 24,
      },
    },
  ],
});
```

Native callers pass the EP option `storageBufferBucketLimits` as a
comma-separated string of decimal `size:limit` pairs, such as
`4194304:24,5242880:24,6291456:24,8388608:24`. The equivalent session
configuration entry is
`ep.webgpuexecutionprovider.storageBufferBucketLimits`.

- Sizes are positive multiples of 16 bytes. Limits are non-negative integers.
  A zero limit preserves the allocation size but retains no free buffers.
- The non-empty table replaces the entire default storage table; omitted sizes
  are not inherited. Pair order does not matter; duplicate sizes are rejected.
- Requests larger than the largest bucket are aligned to 16 bytes and are not
  cached. Other requests use the smallest bucket that fits.
- Requires `storageBufferCacheMode: 'bucket'` (the default). Uniform, query and
  other buffer caches keep their existing settings.
- Set the option when creating the first session on a device. Sessions sharing
  that device share the cache. It does not configure graph capture or JSEP.

Choose limits from the workload's peak demand and available memory. More spares
reduce repeated allocation but retain more memory: the table above can retain
up to 552 MiB of free storage buffers in addition to active allocations. Finer
sizes reduce rounding overhead, but free buffers in earlier buckets remain
cached until the context is released. This option does not add automatic
eviction or change the default table.
