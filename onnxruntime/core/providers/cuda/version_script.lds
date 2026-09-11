#_init and _fini should be local
VERS_1.0 {
  global:
    GetProvider;
    CreateEpFactories;
    ReleaseEpFactory;
    _binary_*;

  # Hide everything else.
  local:
    *;
};
