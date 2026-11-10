VERS_1.0.0 {
 global:
  CreateEpFactories;
  ReleaseEpFactory;
  ExampleEpTestHooks_ResetSyncCount;
  ExampleEpTestHooks_GetSyncCount;
  ExampleEpTestHooks_ResetPreallocatedOutputQuery;
  ExampleEpTestHooks_GetPreallocatedOutputQueryResult;
  ExampleEpTestHooks_GetPreallocatedOutputBadIndexRejected;
  ExampleEpTestHooks_SetCreateDataTransferFailure;
  ExampleEpTestHooks_ResetStateful;
  ExampleEpTestHooks_GetStatefulCalls;
  ExampleEpTestHooks_SetStatefulFailure;
  ExampleEpTestHooks_SetStatefulObserver;
  ExampleEpTestHooks_GetLegacySize;
 local:
    *;
};
