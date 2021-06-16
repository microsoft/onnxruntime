1. Files: nnapi_implementation.h
          nnapi_implementation.cc
          NeuralNetworksTypes.h
These files are copied from TensorFlow Lite project
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/nnapi

These files do not need to be updated frequently, unless new functionalities are
introduced in new Android OS versions, and we will integrate the new functionalities.

The modifications to these files,
NeuralNetworksTypes.h
* The enum ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES was added.
* The operation ANEURALNETWORKS_GROUPED_CONV_2D was added.
nnapi_implementation.h/cc
* CreateNnApiFromSupportLibrary was removed
[TODO, add support of CreateNnApiFromSupportLibrary for Android 12]

2. Files: NeuralNetworksWrapper.h
          NeuralNetworksWrapper.cc
This NeuralNetworksWrapper.h is copied from The Android Open Source Project
https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/runtime/include/
This file is heavily modified, and an extra NeuralNetworksWrapper.cc was added.
Please do not update these files.
