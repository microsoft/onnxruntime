1. Files: nnapi_implementation.h
          nnapi_implementation.cc
          NeuralNetworksTypes.h
These files are copied from TensorFlow Lite project (release tag v2.9.1)
https://github.com/tensorflow/tensorflow/tree/v2.9.1/tensorflow/lite/nnapi

These files do not need to be updated frequently, unless new functionalities are
introduced in new Android OS versions, and we will integrate the new functionalities.

The modifications to these files,
    NeuralNetworksTypes.h
        * The enum ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES was added.
    nnapi_implementation.h/cc
        * Include paths were updated
            * In nnapi_implementation.cc, NeuralNetworksSupportLibraryImpl.h was replaced with NeuralNetworksTypes.h.
              We don't keep a copy of NeuralNetworksSupportLibraryImpl.h yet.
        * CreateNnApiFromSupportLibrary was removed
        * [TODO, add support of CreateNnApiFromSupportLibrary for Android 12]

2. Files: NeuralNetworksWrapper.h
          NeuralNetworksWrapper.cc
This NeuralNetworksWrapper.h is copied from The Android Open Source Project
    * https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/runtime/include/
This file is heavily modified, and an extra NeuralNetworksWrapper.cc was added.
Please do not update these files.

3. File: nnapi_implementation_stub.cc
A stub implementation of nnapi_implementation.h was added to enable unit testing on non-Android platforms.
