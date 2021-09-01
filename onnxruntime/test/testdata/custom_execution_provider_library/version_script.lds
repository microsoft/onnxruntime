#_init and _fini should be local
VERS_1.0 {
  global:
    ProviderEntryPoint;    

  # Hide everything else.
  local:
    *;
};