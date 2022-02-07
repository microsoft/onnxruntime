# Core ML Model Format Specification
This directory contains the protobuf message definitions that comprise the Core ML model document (``.mlmodel``) format.

The top-level message is ``Model``, which is defined in ``Model.proto``.
Other message types describe data structures, feature types, feature engineering model types, and predictive model types.

# Update the Core ML Model Format Specification
Please do not modify protobuf message definitions, they are copied directly from [Core ML Tools](https://github.com/apple/coremltools) repository.

To update the Core ML Model Format Schema schema files to a more recent version:
1. Delete all the protobuf message definitions (`.proto`) from this directory.
2. Copy the new version of protobuf message definitions (`.proto`) from the `mlmodel/format/` directory of preferred coremltools release branch.

# Core ML Model Format Schema version history
## [coremltools 4.0](https://github.com/apple/coremltools/releases/tag/4.0)
[Core ML Model Format Specification](https://github.com/apple/coremltools/tree/4.0/mlmodel/format)
