/**
 * @file flash_namespace_config.h
 * @brief Configuration file for Flash namespace management and isolation
 */
#pragma once

#ifndef FLASH_NAMESPACE_CONFIG_H
#define FLASH_NAMESPACE_CONFIG_H

// Set default namespace to onnxruntime::flash
#ifndef FLASH_NAMESPACE
#define FLASH_NAMESPACE onnxruntime::flash
#endif

#define FLASH_NAMESPACE_ALIAS(name) FLASH_NAMESPACE::name

#define FLASH_NAMESPACE_SCOPE(content) \
  namespace onnxruntime {              \
  namespace flash {                    \
  content                              \
  }                                    \
  }

#endif  // FLASH_NAMESPACE_CONFIG_H
