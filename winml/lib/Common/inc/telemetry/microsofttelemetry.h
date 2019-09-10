// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

/* ++

Module Name:

    MicrosoftTelemetry.h

Abstract:

    Microsoft Telemetry-specific definitions and extensions to logging functions:
        - Opt-in helpers to Microsoft Telemetry (TraceLoggingOptionMicrosoftTelemetry)
        - Keywords for categories (applies to TraceLogging and manifested events)
        - Event tags to influence persistence and latency
        - Field tags to influence PII treatment
        - Privacy data tag types

    These should be used only by ETW providers in the Microsoft Telemetry provider group {4f50731a-89cf-4782-b3e0-dce8c90476ba}.

    Please see the following link for the full specification text:
    https://microsoft.sharepoint.com/teams/osg_threshold_specs/_layouts/15/WopiFrame.aspx?sourcedoc={8e8236cf-6b80-4e3c-9d7a-c35b52588946}&action=view

Environment:

    User mode or kernel mode.

Note:

    If you are seeing unexpected references to EtwSetInformation (kernel-mode)
    or EventSetInformation (user-mode), you can use the
    MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION and
    MICROSOFTTELEMETRY_EVENT_SET_INFORMATION macros to adjust how the
    EnableManifestedProviderForMicrosoftTelemetry function accesses this API.
--*/

#ifndef _MICROSOFTTELEMETRY_
#define _MICROSOFTTELEMETRY_

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/*
Macro TraceLoggingOptionMicrosoftTelemetry():
Wrapper macro for use in TRACELOGGING_DEFINE_PROVIDER that declares the
provider's membership in the Microsoft Telemetry provider group
{4f50731a-89cf-4782-b3e0-dce8c90476ba}. Membership in this group means that
events with keyword MICROSOFT_KEYWORD_TELEMETRY, MICROSOFT_KEYWORD_MEASURES,
or MICROSOFT_KEYWORD_CRITICAL_DATA will be recognized as "telemetry" events by
UTC.

    TraceLoggingOptionMicrosoftTelemetry()

is equivalent to:

    TraceLoggingOptionGroup(0x4f50731a, 0x89cf, 0x4782, 0xb3, 0xe0, 0xdc, 0xe8, 0xc9, 0x4, 0x76, 0xba).

Example:

    TRACELOGGING_DEFINE_PROVIDER(g_hMyProvider, "MyProvider",
        (0xb3864c38, 0x4273, 0x58c5, 0x54, 0x5b, 0x8b, 0x36, 0x08, 0x34, 0x34, 0x71),
        TraceLoggingOptionMicrosoftTelemetry());
*/
#define TraceLoggingOptionMicrosoftTelemetry() \
    TraceLoggingOptionGroup(0x4f50731a, 0x89cf, 0x4782, 0xb3, 0xe0, 0xdc, 0xe8, 0xc9, 0x4, 0x76, 0xba)

/*
Macro TraceLoggingOptionWindowsCoreTelemetry():
Wrapper macro for use in TRACELOGGING_DEFINE_PROVIDER that declares the
provider's membership in the Windows Core Telemetry provider group
{c7de053a-0c2e-4a44-91a2-5222ec2ecdf1}. Membership in this group means that
events with keyword MICROSOFT_KEYWORD_CRITICAL_DATA or event tag
MICROSOFT_EVENTTAG_CORE_DATA will be recognized as "telemetry" events by
UTC even at the Basic level.

    TraceLoggingOptionWindowsCoreTelemetry()

is equivalent to:

    TraceLoggingOptionGroup(0xc7de053a, 0x0c2e, 0x4a44, 0x91, 0xa2, 0x52, 0x22, 0xec, 0x2e, 0xcd, 0xf1).

Example:

    TRACELOGGING_DEFINE_PROVIDER(g_hMyProvider, "MyProvider",
        (0xb3864c38, 0x4273, 0x58c5, 0x54, 0x5b, 0x8b, 0x36, 0x08, 0x34, 0x34, 0x71),
        TraceLoggingOptionWindowsCoreTelemetry());
*/
#define TraceLoggingOptionWindowsCoreTelemetry() \
    TraceLoggingOptionGroup(0xc7de053a, 0x0c2e, 0x4a44, 0x91, 0xa2, 0x52, 0x22, 0xec, 0x2e, 0xcd, 0xf1)

/*
Privacy data tagging: Use TelemetryPrivacyDataTag(tag) in a telemetry
TraceLoggingWrite macro to indicate the type of data being collected.
Use the PDT macro values. If necessary, multiple tags may be OR'ed together,
e.g. TelemetryPrivacyDataTag(PDT_BrowsingHistory | PDT_ProductAndServiceUsage).

Typical usage:

    TraceLoggingWrite(
        g_hMyProviderHandle,
        "MyPerformanceEvent",
        TelemetryPrivacyDataTag(PDT_ProductAndServicePerformance),
        TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
        TraceLoggingValue(MyPerformanceData));

Common PDT macros are defined below. Additional macros for advanced scenarios
are defined in MicrosoftTelemetryPrivacy.h.
*/
#define TelemetryPrivacyDataTag(tag) TraceLoggingUInt64((tag), "PartA_PrivTags")

#define PDT_BrowsingHistory                    0x0000000000000002u
#define PDT_DeviceConnectivityAndConfiguration 0x0000000000000800u
#define PDT_InkingTypingAndSpeechUtterance     0x0000000000020000u
#define PDT_ProductAndServicePerformance       0x0000000001000000u
#define PDT_ProductAndServiceUsage             0x0000000002000000u
#define PDT_SoftwareSetupAndInventory          0x0000000080000000u

#ifndef MICROSOFTTELEMETRY_NO_FUNCTIONS

#include <evntprov.h>

/*
Macro MICROSOFTTELEMETRY_EVENT_SET_INFORMATION:
Macro MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION:

These macros affect the behavior of
EnableManifestedProviderForMicrosoftTelemetry.

The default behavior of EnableManifestedProviderForMicrosoftTelemetry depends
on the version of Windows being targeted. If the targeted version of Windows
supports the EventSetInformation/EtwSetInformation API, then
EnableManifestedProviderForMicrosoftTelemetry will call the API directly,
leading to a static dependency. If the targeted version of Windows does not
support the API, then EnableManifestedProviderForMicrosoftTelemetry will
attempt to dynamically load the appropriate API (via GetProcAddress or
MmGetSystemRoutineAddress), avoiding the static dependency.

Use MICROSOFTTELEMETRY_EVENT_SET_INFORMATION if you want a custom API to be
called instead of EventSetInformation or EtwSetInformation. If
MICROSOFTTELEMETRY_EVENT_SET_INFORMATION is defined, then
EnableManifestedProviderForMicrosoftTelemetry will invoke
MICROSOFTTELEMETRY_EVENT_SET_INFORMATION(...) instead of calling
EventSetInformation(...) or EtwSetInformation(...). Note that
MICROSOFTTELEMETRY_EVENT_SET_INFORMATION will only take effect if
MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION is unset or is set to 1.

Use MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION if you want to override the
Windows version detection of EnableManifestedProviderForMicrosoftTelemetry.
If MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION is not defined,
EnableManifestedProviderForMicrosoftTelemetry will use windows version macros
to determine whether to static-link or runtime-link the EventSetInformation or
EtwSetInformation API. You can set
MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION to 0, 1, or 2 to override this
versioning logic. Set it to 0 to completely disable the API (always return a
NOT_SUPPORTED error). Set it to 1 to force static-link with the API. Set it to
2 to force dynamic-load of the API.
*/
#ifdef _ETW_KM_
#ifndef   MICROSOFTTELEMETRY_EVENT_SET_INFORMATION
  #define MICROSOFTTELEMETRY_EVENT_SET_INFORMATION EtwSetInformation
  #ifndef   MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION
    #if NTDDI_VERSION < 0x06040000
      #define MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION 2 // Find "EtwSetInformation" via MmGetSystemRoutineAddress
    #else
      #define MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION 1 // Directly invoke EtwSetInformation(...)
    #endif
  #endif
#elif !defined(MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION)
  #define MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION 1 // Directly invoke MICROSOFTTELEMETRY_EVENT_SET_INFORMATION(...)
#endif
#else // _ETW_KM_
#ifndef   MICROSOFTTELEMETRY_EVENT_SET_INFORMATION
  #define MICROSOFTTELEMETRY_EVENT_SET_INFORMATION EventSetInformation
  #ifndef MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION
    #if WINVER < 0x0602 || !defined(EVENT_FILTER_TYPE_SCHEMATIZED)
      #define MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION 2 // Find "EventSetInformation" via GetModuleHandleExW+GetProcAddress
    #else
      #define MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION 1 // Directly invoke EventSetInformation(...)
    #endif
  #endif
#elif !defined(MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION)
  #define MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION 1 // Directly invoke MICROSOFTTELEMETRY_EVENT_SET_INFORMATION(...)
#endif
#endif // _ETW_KM_

__inline
#ifdef _ETW_KM_
NTSTATUS
#else // _ETW_KM_
ULONG
#endif // _ETW_KM_
EventSetInformation_ProviderTraits(
    _In_ REGHANDLE RegHandle,
    _In_count_x_(*(UINT16*)Traits) UCHAR const* Traits
    )
/*++

Routine Description:

    This routine calls EventSetInformation(EventProviderSetTraits) to set the
    provider traits for an ETW provider. For this to work, this must be
    called immediately after the provider has been registered (i.e. immediately
    after the call to EventRegister).

    As a side-effect, this function also notifies ETW that this provider
    properly initializes the EVENT_DATA_DESCRIPTOR::Reserved field in all
    calls to EventWrite/EtwWrite.

Arguments:

    RegHandle - The provider registration handle.
    Traits - A pointer to the traits to register. This is assumed to be a valid
             traits blob, starting with a 16-bit length field.

Return Value:

    User mode: ERROR_SUCCESS on success, error code otherwise.
    Kernel mode: STATUS_SUCCESS on success, error code otherwise.

    The most common error code is ERROR_NOT_SUPPORTED/STATUS_NOT_SUPPORTED,
    which occurs if running on a system that does not support Microsoft
    Telemetry.

--*/
{
#ifdef _ETW_KM_
    NTSTATUS Status = STATUS_NOT_SUPPORTED;
#else // _ETW_KM_
    ULONG Status = ERROR_NOT_SUPPORTED;
#endif // _ETW_KM_

#if MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION == 0 // Immediately return NOT_SUPPORTED.

    (void)RegHandle; // Unreferenced parameter
    (void)Traits; // Unreferenced parameter

#elif MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION == 1

    Status = MICROSOFTTELEMETRY_EVENT_SET_INFORMATION(
        RegHandle,
        (EVENT_INFO_CLASS)2, // EventProviderSetTraits
        (PVOID)Traits,
        *(USHORT*)Traits);

#elif MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION != 2

    #error Invalid value for MICROSOFTTELEMETRY_HAVE_EVENT_SET_INFORMATION. Must be 0, 1, or 2.

#elif defined(_ETW_KM_)

    typedef NTSTATUS(NTAPI* PFEtwSetInformation)(
        _In_ REGHANDLE RegHandle,
        _In_ EVENT_INFO_CLASS InformationClass,
        _In_reads_bytes_opt_(InformationLength) PVOID EventInformation,
        _In_ ULONG InformationLength);
    static UNICODE_STRING strEtwSetInformation = {
        sizeof(L"EtwSetInformation") - 2,
        sizeof(L"EtwSetInformation") - 2,
        L"EtwSetInformation"
    };
#pragma warning(push)
#pragma warning(disable: 4055) // Allow the cast from a PVOID to a PFN
    PFEtwSetInformation pfEtwSetInformation =
        (PFEtwSetInformation)MmGetSystemRoutineAddress(&strEtwSetInformation);
#pragma warning(pop)
    if (pfEtwSetInformation)
    {
        Status = pfEtwSetInformation(
            RegHandle,
            (EVENT_INFO_CLASS)2, // EventProviderSetTraits
            (PVOID)Traits,
            *(USHORT*)Traits);
    }

#else

    HMODULE hEventing = NULL;
    if (GetModuleHandleExW(0, L"api-ms-win-eventing-provider-l1-1-0", &hEventing) ||
        GetModuleHandleExW(0, L"advapi32", &hEventing))
    {
        typedef ULONG(WINAPI* PFEventSetInformation)(
            _In_ REGHANDLE RegHandle,
            _In_ EVENT_INFO_CLASS InformationClass,
            _In_reads_bytes_opt_(InformationLength) PVOID EventInformation,
            _In_ ULONG InformationLength);
        PFEventSetInformation pfEventSetInformation =
            (PFEventSetInformation)GetProcAddress(hEventing, "EventSetInformation");
        if (pfEventSetInformation) {
            Status = pfEventSetInformation(
                RegHandle,
                (EVENT_INFO_CLASS)2, // EventProviderSetTraits
                (PVOID)Traits,
                *(USHORT*)Traits);
        }

        FreeLibrary(hEventing);
    }

#endif

    return Status;
}

__inline
#ifdef _ETW_KM_
NTSTATUS
#else // _ETW_KM_
ULONG
#endif // _ETW_KM_
EnableManifestedProviderForMicrosoftTelemetry(
    _In_ REGHANDLE RegHandle
    )
/*++

Routine Description:

    Manifested providers that want to write telemetry must call this function
    immediately after registration. For example:

        EventRegisterMy_Provider_Name();
        EnableManifestedProviderForMicrosoftTelemetry(My_Provider_NameHandle);

    This routine calls EventSetInformation(EventProviderSetTraits) to declare
    the provider's membership in the Microsoft Telemetry provider group
    {4f50731a-89cf-4782-b3e0-dce8c90476ba}. Membership in this group means
    that events with the following keywords will be recognized as "telemetry"
    events by UTC:

    - MICROSOFT_KEYWORD_TELEMETRY (ms:Telemetry)
    - MICROSOFT_KEYWORD_MEASURES (ms:Measures)
    - MICROSOFT_KEYWORD_CRITICAL_DATA (ms:CriticalData)

    In addition, MICROSOFT_KEYWORD_RESERVED_44 (ms:ReservedKeyword44) is
    reserved for future use as a telemetry keyword and must not be used.

    As a side-effect, this function also notifies ETW that this provider
    properly initializes the EVENT_DATA_DESCRIPTOR::Reserved field in all
    calls to EventWrite/EtwWrite.

Arguments:

    RegHandle - The provider registration handle.

Return Value:

    User mode: ERROR_SUCCESS on success, error code otherwise.
    Kernel mode: STATUS_SUCCESS on success, error code otherwise.

    The most common error code is ERROR_NOT_SUPPORTED/STATUS_NOT_SUPPORTED,
    which occurs if running on a system that does not support Microsoft
    Telemetry.

--*/
{
    static UCHAR const Traits[] = {
        0x16, 0x00, 0x00, 0x13, 0x00, 0x01,
        // {4f50731a-89cf-4782-b3e0-dce8c90476ba}
        0x1a, 0x73, 0x50, 0x4f, 0xcf, 0x89, 0x82, 0x47, 0xb3, 0xe0, 0xdc, 0xe8, 0xc9, 0x04, 0x76, 0xba
    };
    return EventSetInformation_ProviderTraits(RegHandle, Traits);
}

__inline
#ifdef _ETW_KM_
NTSTATUS
#else // _ETW_KM_
ULONG
#endif // _ETW_KM_
EnableManifestedProviderForMicrosoftWlanTelemetry(
    _In_ REGHANDLE RegHandle
    )
/*++

Routine Description:

    Not for normal use! This function is only for use with providers that need
    to use alternate keywords instead of the standard ms:Telemetry,
    ms:Measures, and ms:CriticalData keywords. This function declares that the
    associated provider will use alternate keyword values, not the standard
    ones. Unless you've specifically worked with the UTC team on your manifest,
    DO NOT USE THIS FUNCTION!

    Manifested providers that use the alternate keywords and want to write
    telemetry must call this function immediately after registration. For
    example:

        EventRegisterMy_WLAN_Provider_Name();
        EnableManifestedProviderForMicrosoftWlanTelemetry(
            My_WLAN_Provider_NameHandle);

    This routine calls EventSetInformation(EventProviderSetTraits) to declare
    the provider's membership in the Microsoft WLAN Telemetry provider group
    {976a8310-986e-4640-8bfb-7736ee6d9b65}. Membership in this group means
    that events the following keywords will be recognized as "telemetry"
    events by UTC:

    - 0x20000000 (Telemetry)
    - 0x40000000 (Measures)
    - 0x80000000 (CriticalData)

    In addition, keyword 0x10000000 is reserved for future use as a telemetry
    keyword and must not be used.

    As a side-effect, this function also notifies ETW that this provider
    properly initializes the EVENT_DATA_DESCRIPTOR::Reserved field in all
    calls to EventWrite/EtwWrite.

Arguments:

    RegHandle - The provider registration handle.

Return Value:

    User mode: ERROR_SUCCESS on success, error code otherwise.
    Kernel mode: STATUS_SUCCESS on success, error code otherwise.

    The most common error code is ERROR_NOT_SUPPORTED/STATUS_NOT_SUPPORTED,
    which occurs if running on a system that does not support Microsoft
    Telemetry.

--*/
{
    static UCHAR const Traits[] = {
        0x16, 0x00, 0x00, 0x13, 0x00, 0x01,
        // {976a8310-986e-4640-8bfb-7736ee6d9b65}
        0x10, 0x83, 0x6a, 0x97, 0x6e, 0x98, 0x40, 0x46, 0x8b, 0xfb, 0x77, 0x36, 0xee, 0x6d, 0x9b, 0x65
    };
    return EventSetInformation_ProviderTraits(RegHandle, Traits);
}

#endif // !MICROSOFTTELEMETRY_NO_FUNCTIONS

/*
Telemetry categories that can be assigned as event keywords:

    MICROSOFT_KEYWORD_CRITICAL_DATA: Events that power user experiences or are critical to business intelligence
    MICROSOFT_KEYWORD_MEASURES:      Events for understanding measures and reporting scenarios
    MICROSOFT_KEYWORD_TELEMETRY:     Events for general-purpose telemetry

Only one telemetry category should be assigned per event, though an event may also participate in other non-telemetry keywords.

Some categories (such as CRITICAL_DATA) require formal approval before they can be used. Refer to
https://osgwiki.com/wiki/Common_Schema_Event_Overrides
for details on the requirements and how to start the approval process.
*/

// c.f. WINEVENT_KEYWORD_RESERVED_63-56 0xFF00000000000000 // Bits 63-56 - channel keywords
// c.f. WINEVENT_KEYWORD_*              0x00FF000000000000 // Bits 55-48 - system-reserved keywords
#define MICROSOFT_KEYWORD_CRITICAL_DATA 0x0000800000000000 // Bit 47
#define MICROSOFT_KEYWORD_MEASURES      0x0000400000000000 // Bit 46
#define MICROSOFT_KEYWORD_TELEMETRY     0x0000200000000000 // Bit 45
#define MICROSOFT_KEYWORD_RESERVED_44   0x0000100000000000 // Bit 44 (reserved for future assignment)

/*
For manifested providers, add the following xmlns:ms declaration to the instrumentation
element in the manifest:

<instrumentation
    ...
    xmlns:ms="http://manifests.microsoft.com/win/2004/08/windows/events"
    ...>

Then modify the EVENTS_* settings in your SOURCES file to include:

    EVENTS_INCLUDE_MICROSOFT_TELEMETRY=1

Finally, make sure to call EnableManifestedProviderForMicrosoftTelemetry(Provider_NameHandle)
immediately after registering your provider:

    EventRegisterProvider_Name();
    EnableManifestedProviderForMicrosoftTelemetry(Provider_NameHandle);

Events may then be decorated with ms:CriticalData, ms:Measures, and ms:Telemetry; for example:

    <event
        keywords="ms:Telemetry"
        symbol="HelloWorldEvent"
        value="1"
        />
*/

/*
Event tags that can be assigned to influence how the telemetry client handles events (TraceLogging only):

    MICROSOFT_EVENTTAG_CORE_DATA:                This event contains high-priority "core data".

    MICROSOFT_EVENTTAG_INJECT_XTOKEN:            Inject an Xbox identity token into this event.

    MICROSOFT_EVENTTAG_REALTIME_LATENCY:         Send these events in real time.
    MICROSOFT_EVENTTAG_COSTDEFERRED_LATENCY:     Treat these events like NORMAL_LATENCY until they've been stuck on the device for too long,
                                                    then allow them to upload over costed networks.
    MICROSOFT_EVENTTAG_NORMAL_LATENCY:           Send these events via the preferred connection based on device policy.

    MICROSOFT_EVENTTAG_CRITICAL_PERSISTENCE:     Delete these events last when low on spool space.
    MICROSOFT_EVENTTAG_NORMAL_PERSISTENCE:       Delete these events first when low on spool space.

    MICROSOFT_EVENTTAG_DROP_PII:                 The event's Part A will be reduced.
    MICROSOFT_EVENTTAG_HASH_PII:                 The event's Part A will be obscured.
    MICROSOFT_EVENTTAG_MARK_PII:                 The event's Part A will be kept as-is and routed to a private stream in the backend.
    MICROSOFT_EVENTTAG_DROP_PII_EXCEPT_IP:       The event's Part A will be reduced but the IP address will be stamped on the server.

    MICROSOFT_EVENTTAG_AGGREGATE:                The event should be aggregated by the telemetry client rather than sending each discrete event.

For example:

    TraceLoggingWrite(..., TraceLoggingEventTag(MICROSOFT_EVENTTAG_REALTIME_LATENCY), ...)

Some tags require formal approval before they can be used. Refer to
https://osgwiki.com/wiki/Common_Schema_Event_Overrides
for details on the requirements and how to start the approval process.

Note:
    Only the first 28 bits of the following event tag fields are allowed to be used. The rest will get dropped.
*/
#define MICROSOFT_EVENTTAG_AGGREGATE                0x00010000

#define MICROSOFT_EVENTTAG_DROP_PII_EXCEPT_IP       0x00020000
#define MICROSOFT_EVENTTAG_COSTDEFERRED_LATENCY     0x00040000

#define MICROSOFT_EVENTTAG_CORE_DATA                0x00080000
#define MICROSOFT_EVENTTAG_INJECT_XTOKEN            0x00100000

#define MICROSOFT_EVENTTAG_REALTIME_LATENCY         0x00200000
#define MICROSOFT_EVENTTAG_NORMAL_LATENCY           0x00400000

#define MICROSOFT_EVENTTAG_CRITICAL_PERSISTENCE     0x00800000
#define MICROSOFT_EVENTTAG_NORMAL_PERSISTENCE       0x01000000

#define MICROSOFT_EVENTTAG_DROP_PII                 0x02000000
#define MICROSOFT_EVENTTAG_HASH_PII                 0x04000000
#define MICROSOFT_EVENTTAG_MARK_PII                 0x08000000

/*
Field tags that can be assigned to influence how the telemetry client handles fields and generates
Part A's for the containing event (TraceLogging only):

    MICROSOFT_FIELDTAG_DROP_PII: The field contains PII and should be dropped by the telemetry client.
    MICROSOFT_FIELDTAG_HASH_PII: The field contains PII and should be hashed (obfuscated) prior to uploading.

Note that in order to specify a field tag, a field description must be specified as well, e.g.:

    ..., TraceLoggingWideString(wszUser, "UserName", "User name", MICROSOFT_FIELDTAG_HASH_PII), ...
*/

#define MICROSOFT_FIELDTAG_DROP_PII 0x04000000
#define MICROSOFT_FIELDTAG_HASH_PII 0x08000000

#endif // _MICROSOFTTELEMETRY_