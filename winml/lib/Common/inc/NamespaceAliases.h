// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace winrt::Windows::Foundation {}
namespace wf = ::winrt::Windows::Foundation;

namespace winrt::Windows::Foundation::Collections {}
namespace wfc = ::winrt::Windows::Foundation::Collections;

namespace winrt::Windows::Graphics {}
namespace wg = winrt::Windows::Graphics;

namespace winrt::Windows::Graphics::DirectX {}
namespace wgdx = winrt::Windows::Graphics::DirectX;

namespace winrt::Windows::Graphics::Imaging {}
namespace wgi = ::winrt::Windows::Graphics::Imaging;

namespace winrt::Windows::Media {}
namespace wm = ::winrt::Windows::Media;

namespace winrt::Windows::Storage {}
namespace ws = ::winrt::Windows::Storage;

namespace winrt::Windows::Storage::Streams {}
namespace wss = ::winrt::Windows::Storage::Streams;

#define WINML winrt::WINML_ROOT_NS::AI::MachineLearning
namespace WINML {}
namespace winml = WINML;

#define WINMLP winrt::WINML_ROOT_NS::AI::MachineLearning::implementation
namespace WINMLP {}
namespace winmlp = WINMLP;

#define WINML_EXPERIMENTAL winrt::WINML_ROOT_NS::AI::MachineLearning::Experimental
namespace WINML_EXPERIMENTAL {}
namespace winml_experimental = WINML_EXPERIMENTAL;

#define WINML_EXPERIMENTALP winrt::WINML_ROOT_NS::AI::MachineLearning::Experimental::implementation
namespace WINML_EXPERIMENTALP {}
namespace winml_experimentalp = WINML_EXPERIMENTALP;

namespace _winml::Adapter {}
namespace winmla = ::_winml::Adapter;

namespace _winml::Telemetry {}
namespace _winmlt = ::_winml::Telemetry;

namespace _winml::Imaging {}
namespace _winmli = ::_winml::Imaging;
