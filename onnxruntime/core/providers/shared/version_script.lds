#_init and _fini should be local
VERS_1.0 {
  global:
    Provider_GetHost;    
    Provider_SetHost;

  # Hide everything else.
  local:
    *;
};
