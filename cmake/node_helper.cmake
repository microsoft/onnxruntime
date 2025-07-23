# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.25)

# Function to get NPM path from Node.js path
function(get_npm_path_from_node result_var node_path)
  get_filename_component(NODE_DIR ${node_path} DIRECTORY)
  if(WIN32)
    set(NPM_CLI_CANDIDATE "${NODE_DIR}/npm.cmd")
    if(NOT EXISTS ${NPM_CLI_CANDIDATE})
      set(NPM_CLI_CANDIDATE "${NODE_DIR}/npm")
    endif()
  else()
    set(NPM_CLI_CANDIDATE "${NODE_DIR}/npm")
  endif()

  set(${result_var} ${NPM_CLI_CANDIDATE} PARENT_SCOPE)
endfunction()

# Validator function for Node.js installation (checks both Node.js and NPM versions)
function(validate_nodejs_installation result_var node_path)
  # First validate Node.js version
  execute_process(
    COMMAND ${node_path} --version
    OUTPUT_VARIABLE NODE_VERSION_OUTPUT
    ERROR_VARIABLE NODE_VERSION_ERROR
    RESULT_VARIABLE NODE_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NODE_VERSION_RESULT EQUAL 0)
    # Node version output starts with 'v', e.g., "v20.10.0"
    string(REGEX MATCH "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)" NODE_VERSION_MATCH ${NODE_VERSION_OUTPUT})
    if(NODE_VERSION_MATCH)
      set(NODE_VERSION_MAJOR ${CMAKE_MATCH_1})

      if(NODE_VERSION_MAJOR LESS 20)
        message(STATUS "Node.js at ${node_path} version ${NODE_VERSION_OUTPUT} is too old. Required: >= v20.0.0")
        set(${result_var} FALSE PARENT_SCOPE)
        return()
      endif()

      message(STATUS "Found Node.js at ${node_path} with version: ${NODE_VERSION_OUTPUT}")
    else()
      message(STATUS "Could not parse Node.js version from ${node_path}: ${NODE_VERSION_OUTPUT}")
      set(${result_var} FALSE PARENT_SCOPE)
      return()
    endif()
  else()
    message(STATUS "Failed to get Node.js version from ${node_path}: ${NODE_VERSION_ERROR}")
    set(${result_var} FALSE PARENT_SCOPE)
    return()
  endif()

  # Now validate NPM from the same installation directory
  get_npm_path_from_node(NPM_CLI_CANDIDATE ${node_path})

  if(NOT EXISTS ${NPM_CLI_CANDIDATE})
    get_filename_component(NODE_DIR ${node_path} DIRECTORY)
    message(STATUS "Could not find NPM in the same directory as Node.js: ${NODE_DIR}")
    set(${result_var} FALSE PARENT_SCOPE)
    return()
  endif()

  # Validate NPM version
  execute_process(
    COMMAND ${NPM_CLI_CANDIDATE} --version
    OUTPUT_VARIABLE NPM_VERSION_OUTPUT
    ERROR_VARIABLE NPM_VERSION_ERROR
    RESULT_VARIABLE NPM_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NPM_VERSION_RESULT EQUAL 0)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" NPM_VERSION_MATCH ${NPM_VERSION_OUTPUT})
    if(NPM_VERSION_MATCH)
      set(NPM_VERSION_MAJOR ${CMAKE_MATCH_1})

      if(NPM_VERSION_MAJOR LESS 10)
        message(STATUS "NPM at ${NPM_CLI_CANDIDATE} version ${NPM_VERSION_OUTPUT} is too old. Required: >= 10.0.0")
        set(${result_var} FALSE PARENT_SCOPE)
        return()
      endif()

      message(STATUS "Found NPM at ${NPM_CLI_CANDIDATE} with version: ${NPM_VERSION_OUTPUT}")
      set(${result_var} TRUE PARENT_SCOPE)
    else()
      message(STATUS "Could not parse NPM version from ${NPM_CLI_CANDIDATE}: ${NPM_VERSION_OUTPUT}")
      set(${result_var} FALSE PARENT_SCOPE)
    endif()
  else()
    message(STATUS "Failed to get NPM version from ${NPM_CLI_CANDIDATE}: ${NPM_VERSION_ERROR}")
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Check if both Node.js and NPM are already provided
if((NOT NPM_CLI) OR (NOT NODE_EXECUTABLE))
  # Find node executable with combined Node.js + NPM validation
  find_program(NODE_EXECUTABLE
    NAMES "node.exe" "node"
    DOC "Node.js executable"
    VALIDATOR validate_nodejs_installation
    REQUIRED
  )

  # Set NPM_CLI from the validated Node.js installation
  get_npm_path_from_node(NPM_CLI ${NODE_EXECUTABLE})
  set(NPM_CLI ${NPM_CLI} CACHE FILEPATH "NPM command line client" FORCE)
  message(STATUS "Using Node.js and NPM from the same validated installation:")
  message(STATUS "  Node.js: ${NODE_EXECUTABLE}")
  message(STATUS "  NPM:     ${NPM_CLI}")
endif()
