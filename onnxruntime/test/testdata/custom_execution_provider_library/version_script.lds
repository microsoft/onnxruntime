#_init and _fini should be local
VERS_1.0 {
  global:
    GetProvider;
    ProviderHashFunc;
    RegisterCustomOp;

  # Hide everything else.
  local:
    *;
};
