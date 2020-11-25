//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "TraceSession.h"

namespace {

VOID WINAPI EventRecordCallback(EVENT_RECORD* pEventRecord)
{
    auto session = (TraceSession*) pEventRecord->UserContext;
    auto const& hdr = pEventRecord->EventHeader;

    if (session->startTime_ == 0) {
        session->startTime_ = hdr.TimeStamp.QuadPart;
    }

    auto iter = session->eventHandler_.find(hdr.ProviderId);
    if (iter != session->eventHandler_.end()) {
        auto const& h = iter->second;
        (*h.fn_)(pEventRecord, h.ctxt_);
    }
}

ULONG WINAPI BufferCallback(EVENT_TRACE_LOGFILE* pLogFile)
{
    auto session = (TraceSession*) pLogFile->Context;
    auto shouldStopFn = session->shouldStopProcessingEventsFn_;
    if (shouldStopFn && (*shouldStopFn)()) {
        return FALSE; // break out of ProcessTrace()
    }

    return TRUE; // continue processing events
}

bool OpenLogger(
    TraceSession* session,
    TCHAR const* name,
    bool realtime)
{
    // Open trace
    EVENT_TRACE_LOGFILE loggerInfo = {};
    /* Filled out below based on realtime:
    loggerInfo.LogFileName = nullptr;
    loggerInfo.LoggerName = nullptr;
    */
    loggerInfo.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
    loggerInfo.BufferCallback = BufferCallback;
    loggerInfo.EventRecordCallback = EventRecordCallback;
    loggerInfo.Context = session;
    /* Output members (passed also to BufferCallback()):
    loggerInfo.CurrentTime
    loggerInfo.BuffersRead
    loggerInfo.CurrentEvent
    loggerInfo.LogfileHeader
    loggerInfo.BufferSize
    loggerInfo.Filled
    loggerInfo.IsKernelTrace
    */
    /* Not used:
    loggerInfo.EventsLost
    */

    if (realtime) {
        loggerInfo.LoggerName = const_cast<decltype(loggerInfo.LoggerName)>(name);
        loggerInfo.ProcessTraceMode |= PROCESS_TRACE_MODE_REAL_TIME;
    } else {
        loggerInfo.LogFileName = const_cast<decltype(loggerInfo.LoggerName)>(name);
    }

    session->traceHandle_ = OpenTrace(&loggerInfo);
    if (session->traceHandle_ == INVALID_PROCESSTRACE_HANDLE) {
        fprintf(stderr, "error: failed to open trace");
        auto lastError = GetLastError();
        switch (lastError) {
        case ERROR_INVALID_PARAMETER: fprintf(stderr, " (Logfile is NULL)"); break;
        case ERROR_BAD_PATHNAME:      fprintf(stderr, " (invalid LoggerName)"); break;
        case ERROR_ACCESS_DENIED:     fprintf(stderr, " (access denied)"); break;
        default:                      fprintf(stderr, " (error=%u)", lastError); break;
        }
        fprintf(stderr, ".\n");
        return false;
    }

    // Copy desired state from loggerInfo
    session->frequency_ = loggerInfo.LogfileHeader.PerfFreq.QuadPart;
    return true;
}

}

size_t TraceSession::GUIDHash::operator()(GUID const& g) const
{
    static_assert((sizeof(g) % sizeof(size_t)) == 0, "sizeof(GUID) must be multiple of sizeof(size_t)");
    auto p = (size_t const*) &g;
    auto h = (size_t) 0;
    for (size_t i = 0; i < sizeof(g) / sizeof(size_t); ++i) {
        h ^= p[i];
    }
    return h;
}

bool TraceSession::GUIDEqual::operator()(GUID const& lhs, GUID const& rhs) const
{
    return IsEqualGUID(lhs, rhs) != FALSE;
}

bool TraceSession::AddProvider(GUID providerId, UCHAR level,
                               ULONGLONG matchAnyKeyword, ULONGLONG matchAllKeyword)
{
    auto p = eventProvider_.emplace(std::make_pair(providerId, Provider()));
    if (!p.second) {
        return false;
    }

    auto h = &p.first->second;
    h->matchAny_ = matchAnyKeyword;
    h->matchAll_ = matchAllKeyword;
    h->level_    = level;
    return true;
}

bool TraceSession::AddHandler(GUID providerId, EventHandlerFn handlerFn, void* handlerContext)
{
    auto p = eventHandler_.emplace(std::make_pair(providerId, Handler()));
    if (!p.second) {
        return false;
    }

    auto h = &p.first->second;
    h->fn_ = handlerFn;
    h->ctxt_ = handlerContext;
    return true;
}

bool TraceSession::AddProviderAndHandler(GUID providerId, UCHAR level,
                                         ULONGLONG matchAnyKeyword, ULONGLONG matchAllKeyword,
                                         EventHandlerFn handlerFn, void* handlerContext)
{
    if (!AddProvider(providerId, level, matchAnyKeyword, matchAllKeyword))
        return false;
    if (!AddHandler(providerId, handlerFn, handlerContext)) {
        RemoveProvider(providerId);
        return false;
    }
    return true;
}

bool TraceSession::RemoveProvider(GUID providerId)
{
    if (sessionHandle_ != 0) {
        auto status = EnableTraceEx2(sessionHandle_, &providerId, EVENT_CONTROL_CODE_DISABLE_PROVIDER, 0, 0, 0, 0, nullptr);
        (void) status;
    }

    return eventProvider_.erase(providerId) != 0;
}

bool TraceSession::RemoveHandler(GUID providerId)
{
    return eventHandler_.erase(providerId) != 0;
}

bool TraceSession::RemoveProviderAndHandler(GUID providerId)
{
    return RemoveProvider(providerId) || RemoveHandler(providerId);
}

bool TraceSession::InitializeEtlFile(TCHAR const* inputEtlPath, ShouldStopProcessingEventsFn shouldStopFn)
{
    // Open the trace
    if (!OpenLogger(this, inputEtlPath, false)) {
        Finalize();
        return false;
    }

    // Initialize state
    shouldStopProcessingEventsFn_ = shouldStopFn;
    eventsLostCount_ = 0;
    buffersLostCount_ = 0;
    return true;
}

bool TraceSession::InitializeRealtime(TCHAR const* traceSessionName, ShouldStopProcessingEventsFn shouldStopFn)
{
    // Set up and start a real-time collection session
    memset(&properties_, 0, sizeof(properties_));

    properties_.Wnode.BufferSize = (ULONG) offsetof(TraceSession, sessionHandle_);
  //properties_.Wnode.Guid                 // ETW will create Guid
    properties_.Wnode.ClientContext = 1;   // Clock resolution to use when logging the timestamp for each event
                                           // 1 == query performance counter
    properties_.Wnode.Flags = 0;
  //properties_.BufferSize = 0;
    properties_.MinimumBuffers = 200;
  //properties_.MaximumBuffers = 0;
  //properties_.MaximumFileSize = 0;
    properties_.LogFileMode = EVENT_TRACE_REAL_TIME_MODE;
  //properties_.FlushTimer = 0;
  //properties_.EnableFlags = 0;
    properties_.LogFileNameOffset = 0;
    properties_.LoggerNameOffset = offsetof(TraceSession, loggerName_);

    auto status = StartTrace(&sessionHandle_, traceSessionName, &properties_);
    if (status == ERROR_ALREADY_EXISTS) {
#ifdef _DEBUG
        fprintf(stderr, "warning: trying to start trace session that already exists.\n");
#endif
        status = ControlTrace((TRACEHANDLE) 0, traceSessionName, &properties_, EVENT_TRACE_CONTROL_STOP);
        if (status == ERROR_SUCCESS) {
            status = StartTrace(&sessionHandle_, traceSessionName, &properties_);
        }
    }
    if (status != ERROR_SUCCESS) {
        fprintf(stderr, "error: failed to start trace session (error=%lu).\n", status);
        return false;
    }

    // Enable desired providers
    for (auto const& p : eventProvider_) {
        auto pGuid = &p.first;
        auto const& h = p.second;

        status = EnableTraceEx2(sessionHandle_, pGuid, EVENT_CONTROL_CODE_ENABLE_PROVIDER, h.level_, h.matchAny_, h.matchAll_, 0, nullptr);
        if (status != ERROR_SUCCESS) {
            fprintf(stderr, "error: failed to enable provider {%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}.\n",
                pGuid->Data1, pGuid->Data2, pGuid->Data3, pGuid->Data4[0], pGuid->Data4[1], pGuid->Data4[2],
                pGuid->Data4[3], pGuid->Data4[4], pGuid->Data4[5], pGuid->Data4[6], pGuid->Data4[7]);
            Finalize();
            return false;
        }
    }

    // Open the trace
    if (!OpenLogger(this, traceSessionName, true)) {
        Finalize();
        return false;
    }

    // Initialize state
    shouldStopProcessingEventsFn_ = shouldStopFn;
    eventsLostCount_ = 0;
    buffersLostCount_ = 0;

    return true;
}

void TraceSession::Finalize()
{
    ULONG status = ERROR_SUCCESS;

    if (traceHandle_ != INVALID_PROCESSTRACE_HANDLE) {
        status = CloseTrace(traceHandle_);
        traceHandle_ = INVALID_PROCESSTRACE_HANDLE;
    }

    if (sessionHandle_ != 0) {
        status = ControlTraceW(sessionHandle_, nullptr, &properties_, EVENT_TRACE_CONTROL_STOP);

        while (!eventProvider_.empty()) {
            RemoveProvider(eventProvider_.begin()->first);
        }
        while (!eventHandler_.empty()) {
            RemoveHandler(eventHandler_.begin()->first);
        }

        sessionHandle_ = 0;
    }
}

bool TraceSession::CheckLostReports(uint32_t* eventsLost, uint32_t* buffersLost)
{
    if (sessionHandle_ == 0) {
        *eventsLost = 0;
        *buffersLost = 0;
        return false;
    }

    auto status = ControlTraceW(sessionHandle_, nullptr, &properties_, EVENT_TRACE_CONTROL_QUERY);
    if (status == ERROR_MORE_DATA) {    // The buffer &properties_ is too small to hold all the information
        *eventsLost = 0;                // for the session.  If you don't need the session's property information
        *buffersLost = 0;               // you can ignore this error.
        return false;
    }

    if (status != ERROR_SUCCESS) {
        fprintf(stderr, "error: failed to query trace status (%lu).\n", status);
        *eventsLost = 0;
        *buffersLost = 0;
        return false;
    }

    *eventsLost = properties_.EventsLost - eventsLostCount_;
    *buffersLost = properties_.RealTimeBuffersLost - buffersLostCount_;
    eventsLostCount_ = properties_.EventsLost;
    buffersLostCount_ = properties_.RealTimeBuffersLost;
    return *eventsLost + *buffersLost > 0;
}

