#_init and _fini should be local
VERS_1.0 {
  global:
    GetProvider;
    _binary_*;

  # Hide everything else.
  local:
    *;
};
