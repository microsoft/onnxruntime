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
  ExampleEpTestHooks_GetDlpackCurrentDevice;
  ExampleEpTestHooks_GetDlpackSyncCount;
  ExampleEpTestHooks_GetDlpackReleaseCount;
  ExampleEpTestHooks_ResetDlpack;
  ExampleEpTestHooks_SetDlpackSyncBlocked;
 local:
    *;
};
