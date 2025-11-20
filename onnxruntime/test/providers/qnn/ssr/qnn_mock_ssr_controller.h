// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

class QnnMockSSRController {
 public:
  enum class Timing {
    BackendGetBuildId,
    LogCreate,
    BackendCreate,
    ContextCreate,
    BackendValidateOpConfig,
    GraphCreate,
    GraphRetrieve,
    TensorCreateGraphTensor,
    GraphAddNode,
    GraphFinalize,
    ContextGetBinarySize,
    ContextGetBinary,
    GraphExecute
  };

  static QnnMockSSRController& Instance() {
    static QnnMockSSRController instance;
    return instance;
  }

  void SetTiming(Timing timing) {
    timing_ = timing;
  }

  Timing GetTiming() const {
    return timing_;
  }

 private:
  QnnMockSSRController() : timing_(Timing::GraphExecute) {};
  QnnMockSSRController(const QnnMockSSRController&) = delete;
  QnnMockSSRController& operator=(const QnnMockSSRController&) = delete;

  // The timing to trigger SSR. Default to triggering SSR at graphExecute
  Timing timing_ = Timing::GraphExecute;
};
