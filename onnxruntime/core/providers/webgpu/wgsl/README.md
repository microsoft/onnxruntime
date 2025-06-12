\[WIP]


## String Template for WGSL shader generation

### Overflow of parser passes

PASS 1: preprocessor

- #include <file_name>
- #include "file_name"

- #if-#elif-#else-#endif
  - #if CONDITION1
  - #elif CONDITION2
  - #else
  - #endif

  CONDITIONs must be boolean param. no expressions allowed.


PASS 2: pre-defined replacements

- indices helper
  - input_a.numComponents                         -> "1"
  - input_a.rank                                  -> "2"
  - input_a.offsetToIndices(offset)               -> "offset"                       // (input_a.rank < 2)
                                                  "o2i_input_a(offset)"          // (input_a.rank >= 2)
  - input_a.indicesToOffset(x)                    -> "x"                           // (input_a.rank < 2)
                                                  "i2o_input_a(x)"              // (input_a.rank >= 2)

  - input_a.broadcastedIndicesToOffset(...)       // !do not support
  - input_a.indices                               // !do not support, use `input_a_indices_t` instead

  - input_a.indicesSet(indices_var, idx, value)   -> "indices_var = value"         // (input_a.rank < 2)
                                                  "indices_var[idx] = value"   // (input_a.rank >= 2)

  - input_a.indicesGet(indices_var, idx)          -> "indices_var"                 // (input_a.rank < 2)
                                                  "indices_var[idx]"            // (input_a.rank >= 2)

  - input_a.set(v_args)                           -> "set_input_a(arg0, arg1, ...)"
  - input_a.setByIndices(indices_var, value)      -> "set_input_a_by_indices(indices_var, value)"
  - input_a.setByOffset(offset, value)            -> "input_a[offset] = value"

  - input_a.get(v_args)                           -> "get_input_a(arg0, arg1, ...)"
  - input_a.getByIndices(indices_var)             -> "get_input_a_by_indices(indices_var)"
  - input_a.getByOffset(offset)                   -> "input_a[offset]"

- preserved names
  - MAIN  // "MAIN" is a preserved name for using as the main function header

  - sumVector(v1)                                 -> "(v1[0] + v1[1])"           // (v1 is vec2)
                                                     "(v1[0] + v1[1] + v1[2] + v1[3])" // (v1 is vec4)

  - ...

PASS 3: parameter replacement
- user specified parameters
  - PARAM1                                -> std::to_string(params["PARAM1"])

    PARAMs must be able to stringify (basically number and string). no expressions allowed.
