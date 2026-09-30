/*-------------------------------------------------------------------------------------
 *
 * Copyright (c) Microsoft Corporation
 * Licensed under the MIT license
 *
 *-------------------------------------------------------------------------------------*/

#if !defined(__cplusplus)
    #error C++11 required
#endif

#pragma once

#include <unknwn.h>
#include <d3d12.h>
#include <dstorageerr.h>

#define DSTORAGE_SDK_VERSION 300

interface ID3D12Resource;
interface ID3D12Fence;
interface IDStorageStatusArray;

/// <summary>
/// The priority of a DirectStorage queue.
/// </summary>
enum DSTORAGE_PRIORITY : INT8 {
    DSTORAGE_PRIORITY_LOW      = -1,
    DSTORAGE_PRIORITY_NORMAL   = 0,
    DSTORAGE_PRIORITY_HIGH     = 1,
    DSTORAGE_PRIORITY_REALTIME = 2,

    /// <summary>
    /// The following values can be used for iterating over all priority levels.
    /// </summary>
    DSTORAGE_PRIORITY_FIRST    = DSTORAGE_PRIORITY_LOW,
    DSTORAGE_PRIORITY_LAST     = DSTORAGE_PRIORITY_REALTIME,

    DSTORAGE_PRIORITY_COUNT    = 4
};

/// <summary>
/// The minimum valid queue capacity.
/// </summary>
#define DSTORAGE_MIN_QUEUE_CAPACITY             0x80

/// <summary>
/// The maximum valid queue capacity.
/// </summary>
#define DSTORAGE_MAX_QUEUE_CAPACITY             0x2000

/// <summary>
/// The source type of a DirectStorage request.
/// </summary>
enum DSTORAGE_REQUEST_SOURCE_TYPE : UINT64 {
    /// <summary>
    /// The source of the DirectStorage request is a file.
    /// </summary>
    DSTORAGE_REQUEST_SOURCE_FILE = 0,

    /// <summary>
    /// The source of the DirectStorage request is a block of memory.
    /// </summary>
    DSTORAGE_REQUEST_SOURCE_MEMORY = 1,
};

/// <summary>
/// The destination type of a DirectStorage request.
/// </summary>
enum DSTORAGE_REQUEST_DESTINATION_TYPE : UINT64 {
    /// <summary>
    /// The destination of the DirectStorage request is a block of memory.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_MEMORY = 0,

    /// <summary>
    /// The destination of the DirectStorage request is an ID3D12Resource
    /// that is a buffer.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_BUFFER = 1,

    /// <summary>
    /// The destination of the DirectStorage request is an ID3D12Resource
    /// that is a texture.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION = 2,

    /// <summary>
    /// The destination of the DirectStorage request is an ID3D12Resource
    /// that is a texture that will receive all subresources in a
    /// single request.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES = 3,

    /// <summary>
    /// The destination of the DirectStorage request is an ID3D12Resource
    /// that is tiled.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_TILES = 4,

    /// <summary>
    /// The destination of the DirectStorage request is an ID3D12Resource
    /// that is a texture that will receive the number of subresources
    /// specified in a single request.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE = 5
};

/// <summary>
/// The DSTORAGE_QUEUE_DESC structure contains the properties of a DirectStorage
/// queue for the queue's creation.
/// </summary>
struct DSTORAGE_QUEUE_DESC {
    /// <summary>
    /// The source type of requests that this DirectStorage queue can accept.
    /// </summary>
    DSTORAGE_REQUEST_SOURCE_TYPE SourceType;

    /// <summary>
    /// The maximum number of requests that the queue can hold.
    /// </summary>
    UINT16 Capacity;

    /// <summary>
    /// The priority of the requests in this queue.
    /// </summary>
    DSTORAGE_PRIORITY Priority;

    /// <summary>
    /// Optional name of the queue. Used for debugging.
    /// </summary>
    _In_opt_z_ const CHAR *Name;

    /// <summary>
    /// Optional device to use for writing to destination resources and
    /// performing GPU decompression. The destination resource's device
    /// must match this device.
    ///
    /// This member may be null. If you specify a null device, then the
    /// destination type must be DSTORAGE_REQUEST_DESTINATION_MEMORY.
    /// </summary>
    ID3D12Device* Device;
};

/// <summary>
/// The DSTORAGE_QUEUE_INFO structure contains the properties and current state
/// of a DirectStorage queue.
/// </summary>
struct DSTORAGE_QUEUE_INFO {
    /// <summary>
    /// The DSTORAGE_QUEUE_DESC structure used for the queue's creation.
    /// </summary>
    DSTORAGE_QUEUE_DESC Desc;

    /// <summary>
    /// The number of available empty slots. If a queue is empty, then the number
    /// of empty slots equals capacity - 1. The reserved slot is used to
    /// distinguish between empty and full cases.
    /// </summary>
    UINT16 EmptySlotCount;

    /// <summary>
    /// The number of entries that would need to be enqueued in order to trigger
    /// automatic submission.
    /// </summary>
    UINT16 RequestCountUntilAutoSubmit;
};

/// <summary>
/// The type of compression format used at the decompression stage.
/// Your application can implement custom decompressors, starting from
/// DSTORAGE_CUSTOM_COMPRESSION_0.
/// </summary>
enum DSTORAGE_COMPRESSION_FORMAT : UINT8 {
    /// <summary>
    /// The data is uncompressed.
    /// </summary>
    DSTORAGE_COMPRESSION_FORMAT_NONE     = 0,

    /// <summary>
    /// The data is compressed using the built-in GDEFLATE format.
    /// </summary>
    DSTORAGE_COMPRESSION_FORMAT_GDEFLATE = 1,

    /// <summary>
    /// The data is stored in an application-defined custom format. The
    /// application must use IDStorageCustomDecompressionQueue to implement
    /// custom decompression.  Additional custom compression formats can be
    /// used, for example `(DSTORAGE_CUSTOM_COMPRESSION_0 + 1)`.
    DSTORAGE_CUSTOM_COMPRESSION_0        = 0x80,
};

/// <summary>
/// Options for a DirectStorage request.
/// </summary>
struct DSTORAGE_REQUEST_OPTIONS {
    /// <summary>
    /// DSTORAGE_COMPRESSION_FORMAT indicating how the data is compressed.
    /// </summary>
    DSTORAGE_COMPRESSION_FORMAT CompressionFormat : 8;

    /// <summary>
    /// Reserved fields. Must be 0.
    /// </summary>
    UINT8 Reserved1[7];

    /// <summary>
    /// DSTORAGE_REQUEST_SOURCE_TYPE enum value indicating whether the
    /// source of the request is a file or a block of memory.
    /// </summary>
    DSTORAGE_REQUEST_SOURCE_TYPE SourceType : 1;

    /// <summary>
    /// DSTORAGE_REQUEST_DESTINATION_TYPE enum value indicating the
    /// destination of the request. Block of memory, resource.
    /// </summary>
    DSTORAGE_REQUEST_DESTINATION_TYPE DestinationType : 7;

    /// <summary>
    /// Reserved fields. Must be 0.
    /// </summary>
    UINT64 Reserved : 48;
};

/// <summary>
/// Flags controlling DirectStorage debug layer.
/// </summary>
enum DSTORAGE_DEBUG {
    /// <summary>
    /// DirectStorage debug layer is disabled.
    /// </summary>
    DSTORAGE_DEBUG_NONE                 = 0x00,

    /// <summary>
    /// Print error information to a debugger.
    /// </summary>
    DSTORAGE_DEBUG_SHOW_ERRORS          = 0x01,

    /// <summary>
    /// Trigger a debug break each time an error is detected.
    /// </summary>
    DSTORAGE_DEBUG_BREAK_ON_ERROR       = 0x02,

    /// <summary>
    /// Include object names in ETW events.
    /// </summary>
    DSTORAGE_DEBUG_RECORD_OBJECT_NAMES  = 0x04
};
DEFINE_ENUM_FLAG_OPERATORS(DSTORAGE_DEBUG);

/// <summary>
/// Represents a file to be accessed by DirectStorage.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageFile, IUnknown, "5de95e7b-955a-4868-a73c-243b29f4b8da")
{
    /// <summary>
    /// Closes the file, regardless of the reference count on this object.
    ///
    /// After an IDStorageFile object is closed, it can no longer be used in
    /// DirectStorage requests.  This does not modify the reference count on this
    /// object; Release() must be called as usual.
    /// </summary>
    virtual void STDMETHODCALLTYPE Close() = 0;

    /// <summary>
    /// Retrieves file information for an opened file.
    /// </summary>
    /// <param name="info">Receives the file information.</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE GetFileInformation(_Out_ BY_HANDLE_FILE_INFORMATION* info) = 0;
};

/// <summary>
/// Describes a source for a request with SourceType
/// DSTORAGE_REQUEST_SOURCE_FILE.
/// </summary>
struct DSTORAGE_SOURCE_FILE {
    /// <summary>
    /// The file to perform this read request from.
    /// </summary>
    IDStorageFile* Source;

    /// <summary>
    /// The offset, in bytes, in the file to start the read request at.
    /// </summary>
    UINT64 Offset;

    /// <summary>
    /// Number of bytes to read from the file.
    /// </summary>
    UINT32 Size;
};

/// <summary>
/// Describes the source for a request with SourceType
/// DSTORAGE_REQUEST_SOURCE_MEMORY.
/// </summary>
struct DSTORAGE_SOURCE_MEMORY {
    /// <summary>
    /// Address of the source buffer to be read from.
    /// </summary>
    void const* Source;

    /// <summary>
    /// Number of bytes to read from the source buffer.
    /// </summary>
    UINT32 Size;
};

/// <summary>
/// Describes the destination for a request with DestinationType
/// DSTORAGE_REQUEST_DESTINATION_MEMORY.
/// </summary>
struct DSTORAGE_DESTINATION_MEMORY {
    /// <summary>
    /// Address of the buffer to receive the final result of this request.
    /// </summary>
    void* Buffer;

    /// <summary>
    /// Number of bytes to write to the destination buffer.
    /// </summary>
    UINT32 Size;
};

/// <summary>
/// Describes the destination for a request with DestinationType
/// DSTORAGE_REQUEST_DESTINATION_BUFFER.
/// </summary>
struct DSTORAGE_DESTINATION_BUFFER {
    /// <summary>
    /// Address of the resource to receive the final result of this request.
    /// </summary>
    ID3D12Resource* Resource;

    /// <summary>
    /// The offset, in bytes, in the buffer resource to write into.
    /// </summary>
    UINT64 Offset;

    /// <summary>
    /// Number of bytes to write to the destination buffer.
    /// </summary>
    UINT32 Size;
};

/// <summary>
/// Describes the destination for a request with DestinationType
/// DSTORAGE_REQUEST_DESTINATION_TEXTURE_REGION.
/// </summary>
struct DSTORAGE_DESTINATION_TEXTURE_REGION {
    /// <summary>
    /// Address of the resource to receive the final result of this request.
    /// </summary>
    ID3D12Resource* Resource;

    /// <summary>
    /// Describes the destination texture copy location. The subresource
    /// referred to must be in the D3D12_RESOURCE_STATE_COMMON state.
    /// </summary>
    UINT SubresourceIndex;

    /// <summary>
    /// Coordinates and size of the destination region to copy, in pixels.
    /// </summary>
    D3D12_BOX Region;
};

/// <summary>
/// Describes the destination for a request with DestinationType
/// DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES.
/// </summary>
struct DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES {
    /// <summary>
    /// Address of the resource to receive the final result of this request. The
    /// source is expected to contain full data for all subresources, starting
    /// from FirstSubresource.
    /// </summary>
    ID3D12Resource* Resource;

    /// <summary>
    /// Describes the first subresource of the destination texture copy
    /// location. The subresource referred to must be in the
    /// D3D12_RESOURCE_STATE_COMMON state.
    /// </summary>
    UINT FirstSubresource;
};

/// <summary>
/// Describes the destination for a request with DestinationType
/// DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE.
/// </summary>
struct DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE
{
    /// <summary>
    /// Address of the resource to receive the final result of this request. The
    /// source is expected to contain full data for all subresources, starting
    /// from FirstSubresource.
    /// </summary>
    ID3D12Resource* Resource;

    /// <summary>
    /// Describes the first subresource of the destination texture copy
    /// location. The subresource referred to must be in the
    /// D3D12_RESOURCE_STATE_COMMON state.
    /// </summary>
    UINT FirstSubresource;

    /// <summary>
    /// Describes the number of subresources to copy to the destination
    /// resource starting from the FirstSubresource specified.
    /// </summary>
    UINT NumSubresources;
};

/// <summary>
/// Describes the destination for a request with DestinationType
/// DSTORAGE_REQUEST_DESTINATION_TILES.
/// </summary>
struct DSTORAGE_DESTINATION_TILES {
    /// <summary>
    /// Address of the resource to receive the final result of this request. The
    /// source buffer is expected to contain data arranged as if it were the
    /// source to a CopyTiles call with these parameters.
    /// </summary>
    ID3D12Resource* Resource;

    /// <summary>
    /// The starting coordinates of the tiled region.
    /// </summary>
    D3D12_TILED_RESOURCE_COORDINATE TiledRegionStartCoordinate;

    /// <summary>
    /// The size of the tiled region.
    /// </summary>
    D3D12_TILE_REGION_SIZE TileRegionSize;
};

/// <summary>
/// Describes the source specified for a DirectStorage request. For a request,
/// the value of `request.Options.SourceType` determines which of these union
/// fields is active.
/// </summary>
union DSTORAGE_SOURCE {
    DSTORAGE_SOURCE_MEMORY Memory;
    DSTORAGE_SOURCE_FILE File;
};

/// <summary>
/// Describes the destination for a DirectStorage request.  For a request, the
/// value of `request.Options.DestinationType` determines which of these union
/// fields is active.
/// </summary>
union DSTORAGE_DESTINATION {
    DSTORAGE_DESTINATION_MEMORY Memory;
    DSTORAGE_DESTINATION_BUFFER Buffer;
    DSTORAGE_DESTINATION_TEXTURE_REGION Texture;
    DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES MultipleSubresources;
    DSTORAGE_DESTINATION_TILES Tiles;
    DSTORAGE_DESTINATION_MULTIPLE_SUBRESOURCES_RANGE MultipleSubresourcesRange;
};

/// <summary>
/// Represents a DirectStorage request.
/// </summary>
struct DSTORAGE_REQUEST {
    /// <summary>
    /// Combination of decompression and other options for this request.
    /// </summary>
    DSTORAGE_REQUEST_OPTIONS Options;

    /// <summary>
    /// The source for this request.
    /// </summary>
    DSTORAGE_SOURCE Source;

    /// <summary>
    /// The destination for this request.
    /// </summary>
    DSTORAGE_DESTINATION Destination;

    /// <summary>
    /// The uncompressed size in bytes for the destination for this request.
    /// If the request is not compressed, then this can be left as 0.
    ///
    /// For compressed data, if the destination is memory, then the uncompressed size must
    /// exactly equal the destination size. For other destination types, the uncompressed
    /// size may be greater than the destination size.
    ///
    /// If the destination is to memory or buffer, then the destination size should
    /// be specified in the corresponding struct (for example, DSTORAGE_DESTINATION_MEMORY).
    /// For textures, it's the value of pTotalBytes returned by GetCopyableFootprints.
    /// For tiles, it's 64k * number of tiles.
    /// </summary>
    UINT32 UncompressedSize;

    /// <summary>
    /// An arbitrary UINT64 number used for cancellation matching.
    /// </summary>
    UINT64 CancellationTag;

    /// <summary>
    /// Optional name of the request. Used for debugging. If specified, the
    /// string should be accessible until the request completes.
    /// </summary>
    _In_opt_z_ const CHAR *Name;
};

/// <summary>
/// Flags controlling the behavior of requests enqueued using EnqueueRequests.
/// </summary>
enum DSTORAGE_ENQUEUE_REQUEST_FLAGS : UINT32
{
    /// <summary>
    /// Requests wait on the ID3D12Fence before writing to the destination.
    /// All processing required for the requests before the write can be
    /// completed asynchronously once submitted. This is the default behavior.
    /// </summary>
    DSTORAGE_ENQUEUE_REQUEST_FLAG_NONE = 0,

    /// <summary>
    /// Requests wait on the ID3D12Fence before utilizing the GPU for any of
    /// the requests and before writing to the destination. All processing
    /// required for the requests, except GPU work or writing to the
    /// destination, can be completed asynchronously once submitted.
    /// </summary>
    DSTORAGE_ENQUEUE_REQUEST_FLAG_FENCE_WAIT_BEFORE_GPU_WORK = 1,

    /// <summary>
    /// Requests wait on the ID3D12Fence before reading from the source. No
    /// processing occurs until the ID3D12Fence is set.
    /// </summary>
    DSTORAGE_ENQUEUE_REQUEST_FLAG_FENCE_WAIT_BEFORE_SOURCE_ACCESS = 2
};

/// <summary>
/// The maximum number of characters that will be stored for a request's name.
/// </summary>
#define DSTORAGE_REQUEST_MAX_NAME       64

/// <summary>
/// The type of command that failed, as reported by
/// DSTORAGE_ERROR_FIRST_FAILURE.
/// </summary>
enum DSTORAGE_COMMAND_TYPE {
    DSTORAGE_COMMAND_TYPE_NONE = -1,
    DSTORAGE_COMMAND_TYPE_REQUEST = 0,
    DSTORAGE_COMMAND_TYPE_STATUS = 1,
    DSTORAGE_COMMAND_TYPE_SIGNAL = 2,
    DSTORAGE_COMMAND_TYPE_EVENT = 3,
};

/// <summary>
/// The parameters passed to the EnqueueRequest call, and optional
/// filename if the request is for a file source.
/// </summary>
struct DSTORAGE_ERROR_PARAMETERS_REQUEST {
    /// <summary>
    /// For a file source request, the name of the file the request was
    /// targeted to.
    /// </summary>
    WCHAR Filename[MAX_PATH];

    /// <summary>
    /// The name of the request if one was specified.
    /// </summary>
    CHAR RequestName[DSTORAGE_REQUEST_MAX_NAME];

    /// <summary>
    /// The parameters passed to the EnqueueRequest call.
    /// </summary>
    DSTORAGE_REQUEST Request;
};

/// <summary>
/// The parameters passed to the EnqueueStatus call.
/// </summary>
struct DSTORAGE_ERROR_PARAMETERS_STATUS {
    IDStorageStatusArray* StatusArray;
    UINT32 Index;
};

/// <summary>
/// The parameters passed to the EnqueueSignal call.
/// </summary>
struct DSTORAGE_ERROR_PARAMETERS_SIGNAL {
    ID3D12Fence* Fence;
    UINT64 Value;
};

/// <summary>
/// The parameters passed to the EnqueueSetEvent call.
/// </summary>
struct DSTORAGE_ERROR_PARAMETERS_EVENT
{
    HANDLE Handle;
};

/// <summary>
/// Structure to receive the detailed record of the first failed DirectStorage
/// request.
/// </summary>
struct DSTORAGE_ERROR_FIRST_FAILURE {

    /// <summary>
    /// The HRESULT code of the failure.
    /// </summary>
    HRESULT HResult;

    /// <summary>
    /// Type of the Enqueue command that caused the failure.
    /// </summary>
    DSTORAGE_COMMAND_TYPE CommandType;

    /// <summary>
    /// The parameters passed to the Enqueue call.
    /// </summary>
    union
    {
        DSTORAGE_ERROR_PARAMETERS_REQUEST Request;
        DSTORAGE_ERROR_PARAMETERS_STATUS Status;
        DSTORAGE_ERROR_PARAMETERS_SIGNAL Signal;
        DSTORAGE_ERROR_PARAMETERS_EVENT Event;
    };
};

/// <summary>
/// Structure to receive the detailed record of a failed DirectStorage request.
/// </summary>
struct DSTORAGE_ERROR_RECORD {
    /// <summary>
    /// The number of failed requests in the queue since the last
    /// RetrieveErrorRecord call.
    /// </summary>
    UINT32 FailureCount;

    /// <summary>
    /// Detailed record about the first failed command in the enqueue order.
    /// </summary>
    DSTORAGE_ERROR_FIRST_FAILURE FirstFailure;
};


/// <summary>
/// Defines common staging buffer sizes.
/// </summary>
enum DSTORAGE_STAGING_BUFFER_SIZE : UINT32 {
    /// <summary>
    /// There is no staging buffer.  Use this value to force DirectStorage to
    /// deallocate any memory it has allocated for staging buffers.
    /// <summary>
    DSTORAGE_STAGING_BUFFER_SIZE_0 = 0,

    /// <summary>
    /// The default staging buffer size of 32MB.
    /// </summary>
    DSTORAGE_STAGING_BUFFER_SIZE_32MB = 32 * 1048576,
};


/// <summary>
/// Flags used with GetRequests1 when requesting
/// items from the custom decompression queue.
/// </summary>
enum DSTORAGE_GET_REQUEST_FLAGS : UINT32
{
    /// <summary>
    /// Request entries that use custom decompression formats
    /// >= DSTORAGE_CUSTOM_COMPRESSION_0.
    /// </summary>
    DSTORAGE_GET_REQUEST_FLAG_SELECT_CUSTOM = 0x01,

    /// <summary>
    /// Request entries that use built in compression formats
    /// that DirectStorage understands.
    /// </summary>
    DSTORAGE_GET_REQUEST_FLAG_SELECT_BUILTIN = 0x02,

    /// <summary>
    /// Request all entries. This includes custom decompression and
    /// built-in compressed formats.
    /// </summary>
    DSTORAGE_GET_REQUEST_FLAG_SELECT_ALL = (DSTORAGE_GET_REQUEST_FLAG_SELECT_CUSTOM | DSTORAGE_GET_REQUEST_FLAG_SELECT_BUILTIN)
};
DEFINE_ENUM_FLAG_OPERATORS(DSTORAGE_GET_REQUEST_FLAGS);

/// <summary>
/// Specifies information about a custom decompression request.
/// </summary>
enum DSTORAGE_CUSTOM_DECOMPRESSION_FLAGS : UINT32
{
    /// <summary>
    /// No additional information.
    /// </summary>
    DSTORAGE_CUSTOM_DECOMPRESSION_FLAG_NONE = 0x00,

    /// <summary>
    /// The uncompressed destination buffer is located in an
    /// upload heap, and is marked as WRITE_COMBINED.
    /// </summary>
    DSTORAGE_CUSTOM_DECOMPRESSION_FLAG_DEST_IN_UPLOAD_HEAP = 0x01,
};
DEFINE_ENUM_FLAG_OPERATORS(DSTORAGE_CUSTOM_DECOMPRESSION_FLAGS);

/// <summary>
/// A custom decompression request. Use IDStorageCustomDecompressionQueue to
/// retrieve these requests.
/// </summary>
struct DSTORAGE_CUSTOM_DECOMPRESSION_REQUEST {
    /// <summary>
    /// An identifier provided by DirectStorage. This should be used to
    /// identify the request in DSTORAGE_CUSTOM_DECOMPRESSION_RESULT. This
    /// identifier is unique among uncompleted requests, but may be reused after
    /// a request has completed.
    /// </summary>
    UINT64 Id;

    /// <summary>
    /// The compression format.  This will be >= DSTORAGE_CUSTOM_COMPRESSION_0
    /// if DSTORAGE_CUSTOM_DECOMPRESSION_CUSTOMONLY is used to retrieve requests.
    /// </summary>
    DSTORAGE_COMPRESSION_FORMAT CompressionFormat;

    /// <summary>
    /// Reserved for future use.
    /// </summary>
    UINT8 Reserved[3];

    /// <summary>
    /// Flags containing additional details about the decompression request.
    /// </summary>
    DSTORAGE_CUSTOM_DECOMPRESSION_FLAGS Flags;

    /// <summary>
    /// The size of SrcBuffer in bytes.
    /// </summary>
    UINT64 SrcSize;

    /// <summary>
    /// The compressed source buffer.
    /// </summary>
    void const* SrcBuffer;

    /// <summary>
    /// The size of DstBuffer in bytes.
    /// </summary>
    UINT64 DstSize;

    /// <summary>
    /// The uncompressed destination buffer. SrcBuffer should be decompressed to
    /// DstBuffer.
    /// </summary>
    void* DstBuffer;
};

/// <summary>
/// The result of a custom decompression operation. If the request failed, then
/// the Result code is passed back through the standard DirectStorage
/// status/error reporting mechanism.
/// </summary>
struct DSTORAGE_CUSTOM_DECOMPRESSION_RESULT {
    /// <summary>
    /// The identifier for the request, from DSTORAGE_CUSTOM_DECOMPRESSION_REQUEST.
    /// </summary>
    UINT64 Id;

    /// <summary>
    /// The result of this decompression. S_OK indicates success.
    /// </summary>
    HRESULT Result;
};

/// <summary>
/// A queue of decompression requests. This can be obtained using QueryInterface
/// against the factory. Your application must take requests from this queue,
/// decompress them, and report that decompression is complete. That allows an
/// application to provide its own custom decompression.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageCustomDecompressionQueue, IUnknown, "97179b2f-2c21-49ca-8291-4e1bf4a160df")
{
    /// <summary>
    /// Obtains an event to wait on. This event is set when there are pending
    /// decompression requests.
    /// </summary>
    virtual HANDLE STDMETHODCALLTYPE GetEvent() = 0;

    /// <summary>
    /// Populates the given array of request structs with new pending requests.
    /// Your application must arrange to fulfill all these requests, and then
    /// call SetRequestResults to indicate completion.
    /// <summary>
    virtual HRESULT STDMETHODCALLTYPE GetRequests(
        _In_ UINT32 maxRequests,
        _Out_writes_to_(maxRequests, *numRequests) DSTORAGE_CUSTOM_DECOMPRESSION_REQUEST* requests,
        _Out_ UINT32* numRequests) = 0;

    /// <summary>
    /// Your application calls this to indicate that requests have been
    /// completed.
    /// </summary>
    /// <param name="numResults">The number of results in `results`.</param>
    /// <param name="results">An array of results, the size is specified by
    /// `numResults.`</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE SetRequestResults(
        _In_ UINT32 numResults,
        _In_reads_(numResults) DSTORAGE_CUSTOM_DECOMPRESSION_RESULT* results) = 0;
};


/// <summary>
/// An extension of IDStorageCustomDecompressionQueue that allows an
/// application to retrieve specific types of custom decompression
/// requests from the decompression queue.
/// </summary>
DECLARE_INTERFACE_IID_(
    IDStorageCustomDecompressionQueue1,
    IDStorageCustomDecompressionQueue,
    "0D47C6C9-E61A-4706-93B4-68BFE3F4AA4A")
{
    /// <summary>
    /// Populates the given array of request structs with new pending requests
    /// based on the specified custom decompression request type.
    /// The application must arrange to fulfill all these requests, and then
    /// call SetRequestResults to indicate completion.
    /// <summary>
    virtual HRESULT STDMETHODCALLTYPE GetRequests1(
        _In_ DSTORAGE_GET_REQUEST_FLAGS flags,
        _In_ UINT32 maxRequests,
        _Out_writes_to_(maxRequests, *numRequests) DSTORAGE_CUSTOM_DECOMPRESSION_REQUEST* requests,
        _Out_ UINT32 * numRequests) = 0;
};

/// <summary>
/// Represents the static DirectStorage object used to create DirectStorage
/// queues, open files for DirectStorage access, and other global operations.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageFactory, IUnknown, "6924ea0c-c3cd-4826-b10a-f64f4ed927c1")
{
    /// <summary>
    /// Creates a DirectStorage queue object.
    /// </summary>
    /// <param name="desc">Descriptor to specify the properties of the queue.</param>
    /// <param name="riid">Specifies the DirectStorage queue interface, such as
    /// __uuidof(IDStorageQueue).</param>
    /// <param name="ppv">Receives the new queue created.</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE CreateQueue(const DSTORAGE_QUEUE_DESC *desc, REFIID riid, _COM_Outptr_ void **ppv) = 0;

    /// <summary>
    /// Opens a file for DirectStorage access.
    /// </summary>
    /// <param name="path">Path of the file to be opened.</param>
    /// <param name="riid">Specifies the DirectStorage file interface, such as
    /// __uuidof(IDStorageFile).</param>
    /// <param name="ppv">Receives the new file opened.</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE OpenFile(_In_z_ const WCHAR *path, REFIID riid, _COM_Outptr_ void **ppv) = 0;

    /// <summary>
    /// Creates a DirectStorage status array object.
    /// </summary>
    /// <param name="capacity">Specifies the number of statuses that the array can
    /// hold.</param>
    /// <param name="name">Specifies object's name that will appear in
    //  the ETW events if enabled through the debug layer. This is an optional
    //  parameter.</param>
    /// <param name="riid">Specifies the DirectStorage status interface, such as
    /// __uuidof(IDStorageStatusArray).</param>
    /// <param name="ppv">Receives the new status array object created.</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE CreateStatusArray(UINT32 capacity, _In_opt_ PCSTR name, REFIID riid, _COM_Outptr_ void **ppv) = 0;

    /// <summary>
    /// Sets flags used to control the debug layer.
    /// </summary>
    /// <param name="flags">A set of flags controlling the debug layer.</param>
    virtual void STDMETHODCALLTYPE SetDebugFlags(UINT32 flags) = 0;

    /// <summary>
    /// Sets the size of staging buffer(s) used to temporarily store content loaded
    /// from the storage device before they are decompressed. If only uncompressed
    /// memory sourced queues writing to cpu memory destinations are used, then the
    /// staging buffer may be 0-sized.
    /// </summary>
    /// <param name="size">Size, in bytes, of each staging buffer used
    /// to complete a request.</param>
    ///
    /// <remarks>
    /// The default staging buffer is DSTORAGE_STAGING_BUFFER_SIZE_32MB.
    /// If multiple staging buffers are necessary to complete a request, then each
    /// separate staging buffer is allocated to this staging buffer size.
    ///
    /// If the destination is a GPU resource, then some but not all of the staging
    /// buffers will be allocated from VRAM.
    ///
    /// Requests that exceed the specified size to SetStagingBufferSize will fail.
    /// </remarks>
    virtual HRESULT STDMETHODCALLTYPE SetStagingBufferSize(UINT32 size) = 0;
};

/// <summary>
/// Represents an array of status entries to receive completion results for the
/// read requests before them.
/// </summary>
/// <remarks>
/// A status entry receives completion status for all the requests in the
/// DStorageQueue between where it is enqueued and the previously enqueued
/// status entry. Only when all requests enqueued before the status entry
/// complete (that is, IsComplete for the entry returns true), the status entry
/// can be enqueued again.
/// </remarks>
DECLARE_INTERFACE_IID_(IDStorageStatusArray, IUnknown, "82397587-7cd5-453b-a02e-31379bd64656")
{
    /// <summary>
    /// Returns a Boolean value indicating that all requests enqueued prior to the
    /// specified status entry have completed.
    /// </summary>
    /// <param name="index">Specifies the index of the status entry to retrieve.</param>
    /// <returns>Boolean value indicating completion.</returns>
    /// <remarks>This is equivalent to `GetHResult(index) != E_PENDING`.</remarks>
    virtual bool STDMETHODCALLTYPE IsComplete(UINT32 index) = 0;

    /// <summary>
    /// Returns the HRESULT code of all requests between the specified status
    /// entry and the status entry enqueued before it.
    /// </summary>
    /// <param name="index">Specifies the index of the status entry to retrieve.</param>
    /// <returns>HRESULT code of the requests.</returns>
    /// <remarks>
    /// <list type="bullet">
    /// <item><description>
    /// If any requests have not completed yet, the return value is E_PENDING.
    /// </description></item>
    /// <item><description>
    /// If all requests have completed, and there were failure(s), then the return
    /// value stores the failure code of the first failed request in the enqueue
    /// order.
    /// </description></item>
    /// <item><description>
    /// If all requests have completed successfully, then the return value is S_OK.
    /// </description></item>
    /// </list>
    /// </remarks>
    virtual HRESULT STDMETHODCALLTYPE GetHResult(UINT32 index) = 0;
};

/// <summary>
/// Represents a DirectStorage queue to perform read operations.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageQueue, IUnknown, "cfdbd83f-9e06-4fda-8ea5-69042137f49b")
{
    /// <summary>
    /// Enqueues a read request to the queue. The request remains in the queue
    /// until Submit is called, or until the queue is half full.
    /// If there are no free entries in the queue, then the enqueue operation
    /// blocks until one becomes available.
    /// </summary>
    /// <param name="request">The read request to be queued.</param>
    virtual void STDMETHODCALLTYPE EnqueueRequest(const DSTORAGE_REQUEST *request) = 0;

    /// <summary>
    /// Enqueues a status write. The status write happens when all requests
    /// before the status write entry complete. If there were failure(s)
    /// since the previous status write entry, then the HResult of the enqueued
    /// status entry stores the failure code of the first failed request in the
    /// enqueue order.
    /// If there are no free entries in the queue, then the enqueue operation
    /// blocks until one becomes available.
    /// </summary>
    /// <param name="statusArray">IDStorageStatusArray object.</param>
    /// <param name="index">Index of the status entry in the
    /// IDStorageStatusArray object to receive the status.</param>
    virtual void STDMETHODCALLTYPE EnqueueStatus(IDStorageStatusArray *statusArray, UINT32 index) = 0;

    /// <summary>
    /// Enqueues fence write. The fence write happens when all requests before
    /// the fence entry complete.
    /// If there are no free entries in the queue, then the enqueue operation will
    /// block until one becomes available.
    /// </summary>
    /// <param name="fence">An ID3D12Fence to be written.</param>
    /// <param name="value">The value to write to the fence.</param>
    virtual void STDMETHODCALLTYPE EnqueueSignal(ID3D12Fence *fence, UINT64 value) = 0;

    /// <summary>
    /// Submits all requests enqueued so far to DirectStorage to be executed.
    /// </summary>
    virtual void STDMETHODCALLTYPE Submit() = 0;

    /// <summary>
    /// Attempts to cancel a group of previously enqueued read requests. All
    /// previously enqueued requests whose CancellationTag matches the formula
    /// (CancellationTag & mask) == value will be cancelled.
    /// A cancelled request might or might not complete its original read request.
    /// A cancelled request is not counted as a failure in either
    /// IDStorageStatus or DSTORAGE_ERROR_RECORD.
    /// </summary>
    /// <param name="mask">The mask for the cancellation formula.</param>
    /// <param name="value">The value for the cancellation formula.</param>
    virtual void STDMETHODCALLTYPE CancelRequestsWithTag(UINT64 mask, UINT64 value) = 0;

    /// <summary>
    /// Closes the DirectStorage queue, regardless of the reference count on this
    /// object.
    ///
    /// After the Close function returns, the queue will no longer complete any
    /// more requests, even if some are submitted. This does not modify the
    /// reference count on this object; Release() must be called as usual.
    /// </summary>
    virtual void STDMETHODCALLTYPE Close() = 0;

    /// <summary>
    /// Obtains an event to wait on. When there is any error happening for read
    /// requests in this queue, the event will be signaled, and you may call
    /// RetrieveErrorRecord to retrieve diagnostic information.
    /// </summary>
    /// <returns>HANDLE to an event.</returns>
    virtual HANDLE STDMETHODCALLTYPE GetErrorEvent() = 0;

    /// <summary>
    /// When the error event is signaled, this function can be called to
    /// retrieve a DSTORAGE_ERROR_RECORD. Once the error record is retrieved,
    /// this function should not be called until the next time the error event
    /// is signaled.
    /// </summary>
    /// <param name="record">Receives the error record.</param>
    virtual void STDMETHODCALLTYPE RetrieveErrorRecord(_Out_ DSTORAGE_ERROR_RECORD *record) = 0;

    /// <summary>
    /// Obtains information about the queue. It includes the DSTORAGE_QUEUE_DESC
    /// structure used for the queue's creation as well as the number of empty slots
    /// and number of entries that need to be enqueued to trigger automatic
    /// submission.
    /// </summary>
    /// <param name="info">Receives the queue information.</param>
    virtual void STDMETHODCALLTYPE Query(_Out_ DSTORAGE_QUEUE_INFO *info) = 0;
};

/// <summary>
/// Represents a DirectStorage queue to perform read operations.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageQueue1, IDStorageQueue, "dd2f482c-5eff-41e8-9c9e-d2374b278128")
{
    /// <summary>
    /// Enqueues an operation to set the specified event object to a signaled state.
    /// The event object is set when all requests before it complete.
    /// If there are no free entries in the queue the enqueue operation will
    /// block until one becomes available.
    /// </summary>
    /// <param name="handle">A handle to an event object.</param>
    virtual void STDMETHODCALLTYPE EnqueueSetEvent(HANDLE handle) = 0;
};

/// <summary>
/// Flags returned with GetCompressionSupport that describe the features
/// used by the runtime to decompress content.
/// </summary>
enum DSTORAGE_COMPRESSION_SUPPORT : UINT32
{
    /// <summary>
    /// None
    /// </summary>
    DSTORAGE_COMPRESSION_SUPPORT_NONE = 0x0,

    /// <summary>
    /// Optimized driver support for GPU decompression will be used.
    /// </summary>
    DSTORAGE_COMPRESSION_SUPPORT_GPU_OPTIMIZED = 0x01,

    /// <summary>
    /// Built-in GPU decompression fallback shader will be used.  This can occur if
    /// optimized driver support is not available and the D3D12 device used for this
    /// DirectStorage queue supports the required capabilities.
    /// </summary>
    DSTORAGE_COMPRESSION_SUPPORT_GPU_FALLBACK = 0x02,

    /// <summary>
    /// CPU fallback implementation will be used.
    /// This can occur if:
    /// * Optimized driver support and built-in GPU decompression is not available.
    /// * GPU decompression support has been explicitly disabled using
    ///   DSTORAGE_CONFIGURATION.
    /// * DirectStorage runtime encounters a failure during initialization of its
    ///   GPU decompression system.
    /// </summary>
    DSTORAGE_COMPRESSION_SUPPORT_CPU_FALLBACK = 0x04,

    /// <summary>
    /// Executes work on a compute queue.
    /// </summary>
    DSTORAGE_COMPRESSION_SUPPORT_USES_COMPUTE_QUEUE = 0x08,

    /// <summary>
    /// Executes work on a copy queue.
    /// </summary>
    DSTORAGE_COMPRESSION_SUPPORT_USES_COPY_QUEUE = 0x010,
};
DEFINE_ENUM_FLAG_OPERATORS(DSTORAGE_COMPRESSION_SUPPORT);

/// <summary>
/// Represents a DirectStorage queue to perform read operations.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageQueue2, IDStorageQueue1, "b1c9d643-3a49-44a2-b46f-653649470d18")
{
    /// <summary>
    /// Obtains support information about the queue for a specified compression format.
    /// It includes the chosen path that the DirectStorage runtime will use for decompression.
    /// </summary>
    /// <param name="format">Specifies the compression format to retrieve information
    /// about.</param>
    virtual DSTORAGE_COMPRESSION_SUPPORT STDMETHODCALLTYPE GetCompressionSupport(DSTORAGE_COMPRESSION_FORMAT format) = 0;
};

/// <summary>
/// Represents a DirectStorage queue to perform read operations.
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageQueue3, IDStorageQueue2, "deb54c52-eca8-46b3-82a7-031b72262653")
{
    /// <summary>
    /// Enqueues an array of requests to the queue. The requests will be synchronized
    /// with the specified `ID3D12Fence` and processed after the synchronization point.
    /// </summary>
    /// <param name="requests">A pointer to an array of requests that will be synchronized
    /// with the `ID3D12Fence`.</param>
    /// <param name="numRequests">The number of requests in the array pointed to by
    /// `requests`.</param>
    /// <param name="fence">A pointer to an `ID3D12Fence` that will be used to synchronize
    /// the processing of the requests pointed to by `requests`.</param>
    /// <param name="value">The value the `fence` will wait for. Once the `fence` reaches
    /// the specified `value`, the `requests` will start processing past the
    /// synchronization point.</param>
    /// <param name="flag">A flag that specifies the synchronization point for the
    /// `requests`.</param>
    virtual void STDMETHODCALLTYPE EnqueueRequests(
        _In_reads_(numRequests) const DSTORAGE_REQUEST * requests,
        UINT numRequests,
        _In_opt_ ID3D12Fence* fence,
        UINT64 value,
        DSTORAGE_ENQUEUE_REQUEST_FLAGS flag) = 0;
};

/// <summary>
/// Disables built-in decompression.
///
/// Set NumBuiltInCpuDecompressionThreads in DSTORAGE_CONFIGURATION to
/// this value to disable built-in decompression. No decompression threads
/// will be created and the title is fully responsible for checking
/// the custom decompression queue and pulling off ALL entries to decompress.
/// </summary>
#define DSTORAGE_DISABLE_BUILTIN_CPU_DECOMPRESSION -1

/// <summary>
/// DirectStorage Configuration. Zero initializing this will result in the default values.
/// </summary>
struct DSTORAGE_CONFIGURATION {
    /// <summary>
    /// Sets the number of threads to use for submitting IO operations.
    /// Specifying 0 means use the system's best guess at a good value.
    /// Default == 0.
    /// </summary>
    UINT32 NumSubmitThreads;

    /// <summary>
    /// Sets the number of threads to be used by the DirectStorage runtime to
    /// decompress data using the CPU for built-in compressed formats
    /// that cannot be decompressed using the GPU.
    ///
    /// Specifying 0 means to use the system's best guess at a good value.
    ///
    /// Specifying DSTORAGE_DISABLE_BUILTIN_CPU_DECOMPRESSION means no decompression
    /// threads will be created and the title is fully responsible for checking
    /// the custom decompression queue and pulling off ALL entries to decompress.
    ///
    /// Default == 0.
    /// </summary>
    INT32 NumBuiltInCpuDecompressionThreads;

    /// <summary>
    /// Forces the use of the IO mapping layer, even when running on an
    /// operation system that doesn't require it.  This may be useful during
    /// development, but should be set to the FALSE for release. Default=FALSE.
    /// </summary>
    BOOL ForceMappingLayer;

    /// <summary>
    /// Disables the use of the bypass IO optimization, even if it is available.
    /// This might be useful during development, but should be set to FALSE
    /// for release unless ForceFileBuffering is set to TRUE.
    /// Default == FALSE.
    /// </summary>
    BOOL DisableBypassIO;

    /// <summary>
    /// Disables the reporting of telemetry data when set to TRUE.
    /// Telemetry data is enabled by default in the DirectStorage runtime.
    /// Default == FALSE.
    /// </summary>
    BOOL DisableTelemetry;

    /// <summary>
    /// Disables the use of a decompression metacommand, even if one
    /// is available. This will force the runtime to use the built-in GPU decompression
    /// fallback shader.
    /// This may be useful during development, but should be set to the FALSE
    /// for release. Default == FALSE.
    /// </summary>
    BOOL DisableGpuDecompressionMetacommand;

    /// <summary>
    /// Disables the use of GPU based decompression, even if it is available.
    /// This will force the runtime to use the CPU. Default=FALSE.
    /// </summary>
    BOOL DisableGpuDecompression;
};

/// <summary>
/// DirectStorage Configuration. Zero initializing this will result in the default values.
/// </summary>
struct DSTORAGE_CONFIGURATION1
{
    /// <summary>
    /// Sets the number of threads to use for submitting IO operations.
    /// Specifying 0 means use the system's best guess at a good value.
    /// Default == 0.
    /// </summary>
    UINT32 NumSubmitThreads;

    /// <summary>
    /// Sets the number of threads to be used by the DirectStorage runtime to
    /// decompress data using the CPU for built-in compressed formats
    /// that cannot be decompressed using the GPU.
    ///
    /// Specifying 0 means to use the system's best guess at a good value.
    ///
    /// Specifying DSTORAGE_DISABLE_BUILTIN_CPU_DECOMPRESSION means no decompression
    /// threads will be created and the title is fully responsible for checking
    /// the custom decompression queue and pulling off ALL entries to decompress.
    ///
    /// Default == 0.
    /// </summary>
    INT32 NumBuiltInCpuDecompressionThreads;

    /// <summary>
    /// Forces the use of the IO mapping layer, even when running on an
    /// operation system that doesn't require it.  This may be useful during
    /// development, but should be set to the FALSE for release. Default=FALSE.
    /// </summary>
    BOOL ForceMappingLayer;

    /// <summary>
    /// Disables the use of the bypass IO optimization, even if it is available.
    /// This might be useful during development, but should be set to FALSE
    /// for release unless ForceFileBuffering is set to TRUE.
    /// Default == FALSE.
    /// </summary>
    BOOL DisableBypassIO;

    /// <summary>
    /// Disables the reporting of telemetry data when set to TRUE.
    /// Telemetry data is enabled by default in the DirectStorage runtime.
    /// Default == FALSE.
    /// </summary>
    BOOL DisableTelemetry;

    /// <summary>
    /// Disables the use of a decompression metacommand, even if one
    /// is available. This will force the runtime to use the built-in GPU decompression
    /// fallback shader.
    /// This may be useful during development, but should be set to the FALSE
    /// for release. Default == FALSE.
    /// </summary>
    BOOL DisableGpuDecompressionMetacommand;

    /// <summary>
    /// Disables the use of GPU based decompression, even if it is available.
    /// This will force the runtime to use the CPU. Default=FALSE.
    /// </summary>
    BOOL DisableGpuDecompression;

    /// <summary>
    /// Forces the use of the built-in file caching behaviors supported
    /// within the Windows operating system by not setting
    /// FILE_FLAG_NO_BUFFERING when opening files.
    ///
    /// DisableBypassIO must be set to TRUE when using this option or
    /// E_DSTORAGE_FILEBUFFERING_REQUIRES_DISABLED_BYPASSIO will be returned.
    ///
    /// It is the title's responsibility to know when to use this setting.
    /// This feature should ONLY be enabled for slower HDD drives that will
    /// benefit from the OS file buffering features.
    ///
    /// WARNING: Enabling file buffering on high speed drives may reduce
    /// overall performance when reading from that drive because BypassIO
    /// is also disabled. Default=FALSE.
    /// </summary>
    BOOL ForceFileBuffering;
};

/// <summary>
/// Settings controlling DirectStorage compression codec behavior.
/// </summary>
enum DSTORAGE_COMPRESSION : INT32 {

    /// <summary>
    /// Compress data at a fast rate which may not yield the best
    /// compression ratio.
    /// </summary>
    DSTORAGE_COMPRESSION_FASTEST = -1,

    /// <summary>
    /// Compress data at an average rate with a good compression ratio.
    /// </summary>
    DSTORAGE_COMPRESSION_DEFAULT = 0,

    /// <summary>
    /// Compress data at slow rate with the best compression ratio.
    /// </summary>
    DSTORAGE_COMPRESSION_BEST_RATIO = 1
};

/// <summary>
/// Represents the DirectStorage object for compressing and decompressing the buffers.
///
/// Use DStorageCreateCompressionCodec to get an instance of this.
///
/// </summary>
DECLARE_INTERFACE_IID_(IDStorageCompressionCodec, IUnknown, "84ef5121-9b43-4d03-b5c1-cc34606b262d")
{
    /// <summary>
    /// Compresses a buffer of data using a known compression format at the specifed
    /// compression level.
    /// </summary>
    /// <param name="uncompressedData">Points to a buffer containing uncompressed data.</param>
    /// <param name="uncompressedDataSize">Size, in bytes, of the uncompressed data buffer.</param>
    /// <param name="compressionSetting">Specifies the compression settings to use.</param>
    /// <param name="compressedBuffer">Points to a buffer where compressed data will be
    /// written.</param>
    /// <param name="compressedBufferSize">Size, in bytes, of the buffer which will receive
    /// the compressed data</param>
    /// <param name="compressedDataSize">Size, in bytes, of the actual size written to compressedBuffer</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE CompressBuffer(
        const void* uncompressedData,
        size_t uncompressedDataSize,
        DSTORAGE_COMPRESSION compressionSetting,
        void* compressedBuffer,
        size_t compressedBufferSize,
        size_t* compressedDataSize) = 0;

    /// <summary>
    /// Decompresses data previously compressed using CompressBuffer.
    /// </summary>
    /// <param name="compressedData">Points to a buffer containing compressed data.</param>
    /// <param name="compressedDataSize">Size, in bytes, of the compressed data buffer.</param>
    /// <param name="uncompressedBuffer">Points to a buffer where uncompressed data will be
    /// written.</param>
    /// <param name="uncompressedBufferSize">Size, in bytes, of the buffer which will receive
    /// the uncompressed data</param>
    /// <param name="uncompressedDataSize">Size, in bytes, of the actual size written to uncompressedBuffer</param>
    /// <returns>Standard HRESULT error code.</returns>
    virtual HRESULT STDMETHODCALLTYPE DecompressBuffer(
        const void* compressedData,
        size_t compressedDataSize,
        void* uncompressedBuffer,
        size_t uncompressedBufferSize,
        size_t* uncompressedDataSize) = 0;

    /// <summary>
    /// Returns an upper bound estimated size in bytes required to compress the specified data size.
    /// </summary>
    /// <param name="uncompressedDataSize">Size, in bytes, of the data to be compressed</param>
    virtual size_t STDMETHODCALLTYPE CompressBufferBound(size_t uncompressedDataSize) = 0;
};

extern "C" {

/// <summary>
/// Configures DirectStorage. This must be called before the first call to
/// DStorageGetFactory. If this is not called, then default values are used.
/// </summary>
/// <param name="configuration">Specifies the configuration.</param>
/// <returns>Standard HRESULT error code.  The configuration can only be changed
/// when no queue is created and no files are open,
/// E_DSTORAGE_STAGING_BUFFER_LOCKED is returned if this is not the case.</returns>
HRESULT WINAPI DStorageSetConfiguration(DSTORAGE_CONFIGURATION const* configuration);

/// <summary>
/// Configures DirectStorage. This must be called before the first call to
/// DStorageGetFactory. If this is not called, then default values are used.
/// </summary>
/// <param name="configuration">Specifies the configuration.</param>
/// <returns>Standard HRESULT error code.  The configuration can only be changed
/// when no queue is created and no files are open,
/// E_DSTORAGE_STAGING_BUFFER_LOCKED is returned if this is not the case.</returns>
HRESULT WINAPI DStorageSetConfiguration1(DSTORAGE_CONFIGURATION1 const* configuration);

/// <summary>
/// Returns the static DirectStorage factory object used to create DirectStorage queues,
/// open files for DirectStorage access, and other global operations.
/// </summary>
/// <param name="riid">Specifies the DirectStorage factory interface, such as
/// __uuidof(IDStorageFactory)</param>
/// <param name="ppv">Receives the DirectStorage factory object.</param>
/// <returns>Standard HRESULT error code.</returns>
HRESULT WINAPI DStorageGetFactory(REFIID riid, _COM_Outptr_ void** ppv);

/// <summary>
/// Returns an object used to compress/decompress content.
/// Compression codecs are not thread safe so multiple
/// instances are required if the codecs need to be used
/// by multiple threads.
/// </summary>
/// <param name="format">Specifies how the data is compressed.</param>
/// <param name="numThreads">Specifies maximum number of threads this codec
/// will use. Specifying 0 means to use the system's best guess at a good value.</param>
/// <param name="riid">Specifies the DirectStorage compressor/decompressor interface, such as
/// __uuidof(IDStorageCompressionCodec)</param>
/// <param name="ppv">Receives the DirectStorage object.</param>
/// <returns>Standard HRESULT error code.</returns>
HRESULT WINAPI DStorageCreateCompressionCodec(DSTORAGE_COMPRESSION_FORMAT format, UINT32 numThreads, REFIID riid, _COM_Outptr_ void** ppv);

}
