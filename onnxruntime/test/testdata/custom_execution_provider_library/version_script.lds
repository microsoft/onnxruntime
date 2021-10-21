#_init and _fini should be local
VERS_1.0 {
  global:
    GetProvider;   
    ProviderHashFunc; 

  # Hide everything else.
  local:
    *;
};