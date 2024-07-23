/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
 *
 *    Permission is hereby granted, free of charge, to any person obtaining a
 *    copy of this software and associated documentation files (the "Software"),
 *    to deal in the Software without restriction, including without limitation
 *    the rights to use, copy, modify, merge, publish, distribute, sublicense,
 *    and/or sell copies of the Software, and to permit persons to whom the
 *    Software is furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 *    all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *    DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/
#include "core/framework/compute_capability.h"
#include "core/providers/vsinpu/vsinpu_provider_factory.h"
#include "core/providers/vsinpu/vsinpu_provider_factory_creator.h"
#include "core/providers/vsinpu/vsinpu_execution_provider.h"

namespace onnxruntime {

struct VSINPUProviderFactory : IExecutionProviderFactory {
  VSINPUProviderFactory() {}
  ~VSINPUProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> VSINPUProviderFactory::CreateProvider() {
  onnxruntime::VSINPUExecutionProviderInfo info;
  return std::make_unique<onnxruntime::VSINPUExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_VSINPU() {
  return std::make_shared<onnxruntime::VSINPUProviderFactory>();
}

std::shared_ptr<IExecutionProviderFactory>
VSINPUProviderFactoryCreator::Create() {
  return std::make_shared<onnxruntime::VSINPUProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_VSINPU,
                    _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(
      onnxruntime::VSINPUProviderFactoryCreator::Create());
  return nullptr;
}
