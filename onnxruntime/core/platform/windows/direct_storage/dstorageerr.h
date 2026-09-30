/*-------------------------------------------------------------------------------------
 *
 * Copyright (c) Microsoft Corporation
 * Licensed under the MIT license
 *
 *-------------------------------------------------------------------------------------*/
#pragma once

/*++

 MessageId's 0x0000 - 0x00ff (inclusive) are reserved for DirectStorage.

--*/
//
//  Values are 32 bit values laid out as follows:
//
//   3 3 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1
//   1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1 0
//  +---+-+-+-----------------------+-------------------------------+
//  |Sev|C|R|     Facility          |               Code            |
//  +---+-+-+-----------------------+-------------------------------+
//
//  where
//
//      Sev - is the severity code
//
//          00 - Success
//          01 - Informational
//          10 - Warning
//          11 - Error
//
//      C - is the Customer code flag
//
//      R - is a reserved bit
//
//      Facility - is the facility code
//
//      Code - is the facility's status code
//
//
// Define the facility codes
//
#define FACILITY_GAME                    2340


//
// Define the severity codes
//


//
// MessageId: E_DSTORAGE_ALREADY_RUNNING
//
// MessageText:
//
// DStorage is already running exclusively.
//
#define E_DSTORAGE_ALREADY_RUNNING       ((HRESULT)0x89240001L)

//
// MessageId: E_DSTORAGE_NOT_RUNNING
//
// MessageText:
//
// DStorage is not running.
//
#define E_DSTORAGE_NOT_RUNNING           ((HRESULT)0x89240002L)

//
// MessageId: E_DSTORAGE_INVALID_QUEUE_CAPACITY
//
// MessageText:
//
// Invalid queue capacity parameter.
//
#define E_DSTORAGE_INVALID_QUEUE_CAPACITY ((HRESULT)0x89240003L)

//
// MessageId: E_DSTORAGE_XVD_DEVICE_NOT_SUPPORTED
//
// MessageText:
//
// The specified XVD is not on a supported NVMe device.
// This error only applies to Xbox.
//
#define E_DSTORAGE_XVD_DEVICE_NOT_SUPPORTED ((HRESULT)0x89240004L)

//
// MessageId: E_DSTORAGE_UNSUPPORTED_VOLUME
//
// MessageText:
//
// The specified XVD is not on a supported volume.
// This error only applies to Xbox.
//
#define E_DSTORAGE_UNSUPPORTED_VOLUME    ((HRESULT)0x89240005L)

//
// MessageId: E_DSTORAGE_END_OF_FILE
//
// MessageText:
//
// The specified offset and length exceeds the size of the file.
//
#define E_DSTORAGE_END_OF_FILE           ((HRESULT)0x89240007L)

//
// MessageId: E_DSTORAGE_REQUEST_TOO_LARGE
//
// MessageText:
//
// The IO request is too large.
//
#define E_DSTORAGE_REQUEST_TOO_LARGE     ((HRESULT)0x89240008L)

//
// MessageId: E_DSTORAGE_ACCESS_VIOLATION
//
// MessageText:
//
// The destination buffer for the DStorage request is not accessible.
//
#define E_DSTORAGE_ACCESS_VIOLATION      ((HRESULT)0x89240009L)

//
// MessageId: E_DSTORAGE_UNSUPPORTED_FILE
//
// MessageText:
//
// The file is not supported by DStorage. Possible reasons include the file is a
// sparse file, or is compressed in NTFS.
// This error only applies to Xbox.
//
#define E_DSTORAGE_UNSUPPORTED_FILE      ((HRESULT)0x8924000AL)

//
// MessageId: E_DSTORAGE_FILE_NOT_OPEN
//
// MessageText:
//
// The file is not open.
//
#define E_DSTORAGE_FILE_NOT_OPEN         ((HRESULT)0x8924000BL)

//
// MessageId: E_DSTORAGE_RESERVED_FIELDS
//
// MessageText:
//
// A reserved field is not set to 0.
//
#define E_DSTORAGE_RESERVED_FIELDS       ((HRESULT)0x8924000CL)

//
// MessageId: E_DSTORAGE_INVALID_BCPACK_MODE
//
// MessageText:
//
// The request has invalid BCPack decompression mode.
// This error only applies to Xbox.
//
#define E_DSTORAGE_INVALID_BCPACK_MODE   ((HRESULT)0x8924000DL)

//
// MessageId: E_DSTORAGE_INVALID_SWIZZLE_MODE
//
// MessageText:
//
// The request has invalid swizzle mode.
// This error only applies to Xbox.
//
#define E_DSTORAGE_INVALID_SWIZZLE_MODE  ((HRESULT)0x8924000EL)

//
// MessageId: E_DSTORAGE_INVALID_DESTINATION_SIZE
//
// MessageText:
//
// The request's destination size is invalid. If no decompression is used, it must
// be equal to the request's length; If decompression is used, it must be larger
// than the request's length.
//
#define E_DSTORAGE_INVALID_DESTINATION_SIZE ((HRESULT)0x8924000FL)

//
// MessageId: E_DSTORAGE_QUEUE_CLOSED
//
// MessageText:
//
// The request targets a queue that is closed.
//
#define E_DSTORAGE_QUEUE_CLOSED          ((HRESULT)0x89240010L)

//
// MessageId: E_DSTORAGE_INVALID_CLUSTER_SIZE
//
// MessageText:
//
// The volume is formatted with an unsupported cluster size.
// This error only applies to Xbox.
//
#define E_DSTORAGE_INVALID_CLUSTER_SIZE  ((HRESULT)0x89240011L)

//
// MessageId: E_DSTORAGE_TOO_MANY_QUEUES
//
// MessageText:
//
// The number of queues has reached the maximum limit.
//
#define E_DSTORAGE_TOO_MANY_QUEUES       ((HRESULT)0x89240012L)

//
// MessageId: E_DSTORAGE_INVALID_QUEUE_PRIORITY
//
// MessageText:
//
// Invalid priority is specified for the queue.
//
#define E_DSTORAGE_INVALID_QUEUE_PRIORITY ((HRESULT)0x89240013L)

//
// MessageId: E_DSTORAGE_TOO_MANY_FILES
//
// MessageText:
//
// The number of files has reached the maximum limit.
//
#define E_DSTORAGE_TOO_MANY_FILES        ((HRESULT)0x89240014L)

//
// MessageId: E_DSTORAGE_INDEX_BOUND
//
// MessageText:
//
// The index parameter is out of bound.
//
#define E_DSTORAGE_INDEX_BOUND           ((HRESULT)0x89240015L)

//
// MessageId: E_DSTORAGE_IO_TIMEOUT
//
// MessageText:
//
// The IO operation has timed out.
//
#define E_DSTORAGE_IO_TIMEOUT            ((HRESULT)0x89240016L)

//
// MessageId: E_DSTORAGE_INVALID_FILE_HANDLE
//
// MessageText:
//
// The specified file has not been opened.
//
#define E_DSTORAGE_INVALID_FILE_HANDLE   ((HRESULT)0x89240017L)

//
// MessageId: E_DSTORAGE_DEPRECATED_PREVIEW_GDK
//
// MessageText:
//
// This GDK preview is deprecated. Update to a supported GDK version.
// This error only applies to Xbox.
//
#define E_DSTORAGE_DEPRECATED_PREVIEW_GDK ((HRESULT)0x89240018L)

//
// MessageId: E_DSTORAGE_XVD_NOT_REGISTERED
//
// MessageText:
//
// The specified XVD is not registered or unmounted.
// This error only applies to Xbox.
//
#define E_DSTORAGE_XVD_NOT_REGISTERED    ((HRESULT)0x89240019L)

//
// MessageId: E_DSTORAGE_INVALID_FILE_OFFSET
//
// MessageText:
//
// The request has invalid file offset for the specified decompression mode.
//
#define E_DSTORAGE_INVALID_FILE_OFFSET   ((HRESULT)0x8924001AL)

//
// MessageId: E_DSTORAGE_INVALID_SOURCE_TYPE
//
// MessageText:
//
// A memory source request was enqueued into a file source queue, or a file source
// request was enqueued into a memory source queue.
//
#define E_DSTORAGE_INVALID_SOURCE_TYPE   ((HRESULT)0x8924001BL)

//
// MessageId: E_DSTORAGE_INVALID_INTERMEDIATE_SIZE
//
// MessageText:
//
// The request has invalid intermediate size for the specified decompression modes.
// This error only applies to Xbox.
//
#define E_DSTORAGE_INVALID_INTERMEDIATE_SIZE ((HRESULT)0x8924001CL)

//
// MessageId: E_DSTORAGE_SYSTEM_NOT_SUPPORTED
//
// MessageText:
//
// This console generation doesn't support DirectStorage.
// This error only applies to Xbox.
//
#define E_DSTORAGE_SYSTEM_NOT_SUPPORTED  ((HRESULT)0x8924001DL)

//
// MessageId: E_DSTORAGE_STAGING_BUFFER_LOCKED
//
// MessageText:
//
// Staging buffer size can only be changed when no queue is created and no file is
// open.
//
#define E_DSTORAGE_STAGING_BUFFER_LOCKED ((HRESULT)0x8924001FL)

//
// MessageId: E_DSTORAGE_INVALID_STAGING_BUFFER_SIZE
//
// MessageText:
//
// The specified staging buffer size is not valid.
//
#define E_DSTORAGE_INVALID_STAGING_BUFFER_SIZE ((HRESULT)0x89240020L)

//
// MessageId: E_DSTORAGE_STAGING_BUFFER_TOO_SMALL
//
// MessageText:
//
// The staging buffer isn't large enough to perform this operation.
//
#define E_DSTORAGE_STAGING_BUFFER_TOO_SMALL ((HRESULT)0x89240021L)

//
// MessageId: E_DSTORAGE_INVALID_FENCE
//
// MessageText:
//
// The fence is not valid or has been released.
//
#define E_DSTORAGE_INVALID_FENCE         ((HRESULT)0x89240022L)

//
// MessageId: E_DSTORAGE_INVALID_STATUS_ARRAY
//
// MessageText:
//
// The status array is not valid or has been released.
//
#define E_DSTORAGE_INVALID_STATUS_ARRAY  ((HRESULT)0x89240023L)

//
// MessageId: E_DSTORAGE_INVALID_MEMORY_QUEUE_PRIORITY
//
// MessageText:
//
// Invalid priority is specified for the queue. Only DSTORAGE_PRIORITY_REALTIME
// is a valid priority for a memory queue.
//
#define E_DSTORAGE_INVALID_MEMORY_QUEUE_PRIORITY ((HRESULT)0x89240024L)

//
// MessageId: E_DSTORAGE_DECOMPRESSION_ERROR
//
// MessageText:
//
// A generic error has happened during decompression.
//
#define E_DSTORAGE_DECOMPRESSION_ERROR   ((HRESULT)0x89240030L)

//
// MessageId: E_DSTORAGE_ZLIB_BAD_HEADER
//
// MessageText:
//
// ZLIB header is corrupted.
// This error only applies to Xbox.
//
#define E_DSTORAGE_ZLIB_BAD_HEADER       ((HRESULT)0x89240031L)

//
// MessageId: E_DSTORAGE_ZLIB_BAD_DATA
//
// MessageText:
//
// ZLIB compressed data is corrupted/invalid.
// This error only applies to Xbox.
//
#define E_DSTORAGE_ZLIB_BAD_DATA         ((HRESULT)0x89240032L)

//
// MessageId: E_DSTORAGE_ZLIB_PARITY_FAIL
//
// MessageText:
//
// Block-level ADLER parity check failed during ZLIB decompression.
// This error only applies to Xbox.
//
#define E_DSTORAGE_ZLIB_PARITY_FAIL      ((HRESULT)0x89240033L)

//
// MessageId: E_DSTORAGE_BCPACK_BAD_HEADER
//
// MessageText:
//
// BCPack header is corrupted.
// This error only applies to Xbox.
//
#define E_DSTORAGE_BCPACK_BAD_HEADER     ((HRESULT)0x89240034L)

//
// MessageId: E_DSTORAGE_BCPACK_BAD_DATA
//
// MessageText:
//
// BCPack decoder has generated more data than expected, most likely due to
// corrupted bitstream.
// This error only applies to Xbox.
//
#define E_DSTORAGE_BCPACK_BAD_DATA       ((HRESULT)0x89240035L)

//
// MessageId: E_DSTORAGE_DECRYPTION_ERROR
//
// MessageText:
//
// A generic error has happened during decryption.
// This error only applies to Xbox.
//
#define E_DSTORAGE_DECRYPTION_ERROR      ((HRESULT)0x89240036L)

//
// MessageId: E_DSTORAGE_PASSTHROUGH_ERROR
//
// MessageText:
//
// A generic error has happened during copy operation.
// This error only applies to Xbox.
//
#define E_DSTORAGE_PASSTHROUGH_ERROR     ((HRESULT)0x89240037L)

//
// MessageId: E_DSTORAGE_FILE_TOO_FRAGMENTED
//
// MessageText:
//
// The file is too fragmented to be accessed by DStorage. This error can only
// happen with files overly fragmented on a writable volume.
// This error only applies to Xbox.
//
#define E_DSTORAGE_FILE_TOO_FRAGMENTED   ((HRESULT)0x89240038L)

//
// MessageId: E_DSTORAGE_COMPRESSED_DATA_TOO_LARGE
//
// MessageText:
//
// The size of the resulting compressed data is too large for
// DirectStorage to decompress successfully on the GPU.
//
#define E_DSTORAGE_COMPRESSED_DATA_TOO_LARGE ((HRESULT)0x89240039L)

//
// MessageId: E_DSTORAGE_INVALID_DESTINATION_TYPE
//
// MessageText:
//
// A gpu memory destination request was enqueued into a queue that
// was created without a D3D device or the destination type is 
// unknown.
//
#define E_DSTORAGE_INVALID_DESTINATION_TYPE ((HRESULT)0x89240040L)

//
// MessageId: E_DSTORAGE_FILEBUFFERING_REQUIRES_DISABLED_BYPASSIO
//
// MessageText:
//
// ForceFileBuffering was enabled without disabling BypassIO.
//
#define E_DSTORAGE_FILEBUFFERING_REQUIRES_DISABLED_BYPASSIO ((HRESULT)0x89240041L)
