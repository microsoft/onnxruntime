#_init and _fini should be local
VERS_1.0 {
  global:
    GetProvider;    

  # Hide everything else.
  local:
    *;
};