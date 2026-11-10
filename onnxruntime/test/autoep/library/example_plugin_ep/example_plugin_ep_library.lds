VERS_1.0.0 {
 global:
  CreateEpFactories;
  ReleaseEpFactory;
  ExampleEpTestHooks_GetEpCacheOutstandingBufferCount;
  ExampleEpTestHooks_ResetEpCacheCounts;
  ExampleEpTestHooks_GetEpCachePlanCreatedCount;
  ExampleEpTestHooks_GetEpCachePlanReusedCount;
  ExampleEpTestHooks_ResetSyncCount;
  ExampleEpTestHooks_GetSyncCount;
  ExampleEpTestHooks_ResetPreallocatedOutputQuery;
  ExampleEpTestHooks_GetPreallocatedOutputQueryResult;
  ExampleEpTestHooks_GetPreallocatedOutputBadIndexRejected;
  ExampleEpTestHooks_SetCreateDataTransferFailure;
 local:
    *;
};
