// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    class ICommandRecorder
    {
    public:
        virtual ~ICommandRecorder() = default;

        virtual void Open() = 0;

        // Forces all queued work to begin executing on the GPU. This method returns immediately and does not wait
        // for the submitted work to complete execution on the GPU.
        virtual void CloseAndExecute() = 0;

        virtual bool HasUnsubmittedWork() = 0;
    };

} // namespace Dml
