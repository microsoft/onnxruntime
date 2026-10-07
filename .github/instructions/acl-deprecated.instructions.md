---
description: "Review guidance for changes to the deprecated ACL execution provider and its eventual removal."
applyTo: "onnxruntime/core/providers/acl/**,include/onnxruntime/core/providers/acl/**,cmake/onnxruntime_providers_acl.cmake"
---

# Deprecated ACL Execution Provider

The ACL EP is deprecated and will be removed in a future release.

## Review Policy

Report changes that expand or prolong the deprecated ACL EP surface and require explicit maintainer justification
that the scope is appropriate.

Accept changes that directly remove code or enable removal.
