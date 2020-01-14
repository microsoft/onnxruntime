// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "eparser.h"

// Get the metadata for the event.

// Get the length of the property data. For MOF-based events, the size is inferred from the data type
// of the property. For manifest-based events, the property can specify the size of the property value
// using the length attribute. The length attribue can specify the size directly or specify the name
// of another property in the event data that contains the size. If the property does not include the
// length attribute, the size is inferred from the data type. The length will be zero for variable
// length, null-terminated strings and structures.

DWORD GetPropertyLength(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, USHORT i, PUSHORT PropertyLength);

// Get the size of the array. For MOF-based events, the size is specified in the declaration or using
// the MAX qualifier. For manifest-based events, the property can specify the size of the array
// using the count attribute. The count attribue can specify the size directly or specify the name
// of another property in the event data that contains the size.

DWORD GetArraySize(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, USHORT i, PUSHORT ArraySize);

// Both MOF-based events and manifest-based events can specify name/value maps. The
// map values can be integer values or bit values. If the property specifies a value
// map, get the map.

DWORD GetMapInfo(PEVENT_RECORD pEvent, LPWSTR pMapName, DWORD DecodingSource, PEVENT_MAP_INFO& pMapInfo);

// Print the property.
template <typename T>
PBYTE PrintProperties(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, DWORD PointerSize, USHORT i, PBYTE pUserData,
                      PBYTE pEndOfUserData, const T& t) {
  TDHSTATUS status = ERROR_SUCCESS;
  USHORT PropertyLength = 0;
  DWORD FormattedDataSize = 0;
  USHORT UserDataConsumed = 0;
  LPWSTR pFormattedData = NULL;
  DWORD LastMember = 0;  // Last member of a structure
  USHORT ArraySize = 0;
  PEVENT_MAP_INFO pMapInfo = NULL;

  // Get the length of the property.

  status = GetPropertyLength(pEvent, pInfo, i, &PropertyLength);
  if (ERROR_SUCCESS != status) {
    wprintf(L"GetPropertyLength failed.\n");
    pUserData = NULL;
    goto cleanup;
  }

  // Get the size of the array if the property is an array.

  status = GetArraySize(pEvent, pInfo, i, &ArraySize);

  for (USHORT k = 0; k < ArraySize; k++) {
    // If the property is a structure, print the members of the structure.

    if ((pInfo->EventPropertyInfoArray[i].Flags & PropertyStruct) == PropertyStruct) {
      LastMember = pInfo->EventPropertyInfoArray[i].structType.StructStartIndex +
                   pInfo->EventPropertyInfoArray[i].structType.NumOfStructMembers;

      for (USHORT j = pInfo->EventPropertyInfoArray[i].structType.StructStartIndex; j < LastMember; j++) {
        pUserData = PrintProperties(pEvent, pInfo, PointerSize, j, pUserData, pEndOfUserData, t);
        if (NULL == pUserData) {
          wprintf(L"Printing the members of the structure failed.\n");
          pUserData = NULL;
          goto cleanup;
        }
      }
    } else {
      // Get the name/value mapping if the property specifies a value map.

      status =
          GetMapInfo(pEvent, (PWCHAR)((PBYTE)(pInfo) + pInfo->EventPropertyInfoArray[i].nonStructType.MapNameOffset),
                     pInfo->DecodingSource, pMapInfo);

      if (ERROR_SUCCESS != status) {
        wprintf(L"GetMapInfo failed\n");
        pUserData = NULL;
        goto cleanup;
      }

      // Get the size of the buffer required for the formatted data.

      status = TdhFormatProperty(pInfo, pMapInfo, PointerSize, pInfo->EventPropertyInfoArray[i].nonStructType.InType,
                                 pInfo->EventPropertyInfoArray[i].nonStructType.OutType, PropertyLength,
                                 (USHORT)(pEndOfUserData - pUserData), pUserData, &FormattedDataSize, pFormattedData,
                                 &UserDataConsumed);

      if (ERROR_INSUFFICIENT_BUFFER == status) {
        if (pFormattedData) {
          free(pFormattedData);
          pFormattedData = NULL;
        }

        pFormattedData = (LPWSTR)malloc(FormattedDataSize);
        if (pFormattedData == NULL) {
          wprintf(L"Failed to allocate memory for formatted data (size=%lu).\n", FormattedDataSize);
          status = ERROR_OUTOFMEMORY;
          pUserData = NULL;
          goto cleanup;
        }

        // Retrieve the formatted data.

        status = TdhFormatProperty(pInfo, pMapInfo, PointerSize, pInfo->EventPropertyInfoArray[i].nonStructType.InType,
                                   pInfo->EventPropertyInfoArray[i].nonStructType.OutType, PropertyLength,
                                   (USHORT)(pEndOfUserData - pUserData), pUserData, &FormattedDataSize, pFormattedData,
                                   &UserDataConsumed);
      }

      if (ERROR_SUCCESS == status) {
        t((PWCHAR)((PBYTE)(pInfo) + pInfo->EventPropertyInfoArray[i].NameOffset), pFormattedData);
        pUserData += UserDataConsumed;
      } else {
        wprintf(L"TdhFormatProperty failed with %lu.\n", status);
        pUserData = NULL;
        goto cleanup;
      }
    }
  }

cleanup:

  if (pFormattedData) {
    free(pFormattedData);
    pFormattedData = NULL;
  }

  if (pMapInfo) {
    free(pMapInfo);
    pMapInfo = NULL;
  }

  return pUserData;
}

DWORD GetPropertyLength(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, USHORT i, PUSHORT PropertyLength) {
  DWORD status = ERROR_SUCCESS;
  PROPERTY_DATA_DESCRIPTOR DataDescriptor;
  DWORD PropertySize = 0;

  // If the property is a binary blob and is defined in a manifest, the property can
  // specify the blob's size or it can point to another property that defines the
  // blob's size. The PropertyParamLength flag tells you where the blob's size is defined.

  if ((pInfo->EventPropertyInfoArray[i].Flags & PropertyParamLength) == PropertyParamLength) {
    DWORD Length = 0;  // Expects the length to be defined by a UINT16 or UINT32
    DWORD j = pInfo->EventPropertyInfoArray[i].lengthPropertyIndex;
    ZeroMemory(&DataDescriptor, sizeof(PROPERTY_DATA_DESCRIPTOR));
    DataDescriptor.PropertyName = (ULONGLONG)((PBYTE)(pInfo) + pInfo->EventPropertyInfoArray[j].NameOffset);
    DataDescriptor.ArrayIndex = ULONG_MAX;
    status = TdhGetPropertySize(pEvent, 0, NULL, 1, &DataDescriptor, &PropertySize);
    status = TdhGetProperty(pEvent, 0, NULL, 1, &DataDescriptor, PropertySize, (PBYTE)&Length);
    *PropertyLength = (USHORT)Length;
  } else {
    if (pInfo->EventPropertyInfoArray[i].length > 0) {
      *PropertyLength = pInfo->EventPropertyInfoArray[i].length;
    } else {
      // If the property is a binary blob and is defined in a MOF class, the extension
      // qualifier is used to determine the size of the blob. However, if the extension
      // is IPAddrV6, you must set the PropertyLength variable yourself because the
      // EVENT_PROPERTY_INFO.length field will be zero.

      if (TDH_INTYPE_BINARY == pInfo->EventPropertyInfoArray[i].nonStructType.InType &&
          TDH_OUTTYPE_IPV6 == pInfo->EventPropertyInfoArray[i].nonStructType.OutType) {
        *PropertyLength = (USHORT)sizeof(IN6_ADDR);
      } else if (TDH_INTYPE_UNICODESTRING == pInfo->EventPropertyInfoArray[i].nonStructType.InType ||
                 TDH_INTYPE_ANSISTRING == pInfo->EventPropertyInfoArray[i].nonStructType.InType ||
                 (pInfo->EventPropertyInfoArray[i].Flags & PropertyStruct) == PropertyStruct) {
        *PropertyLength = pInfo->EventPropertyInfoArray[i].length;
      } else {
        wprintf(L"Unexpected length of 0 for intype %d and outtype %d\n",
                pInfo->EventPropertyInfoArray[i].nonStructType.InType,
                pInfo->EventPropertyInfoArray[i].nonStructType.OutType);

        status = ERROR_EVT_INVALID_EVENT_DATA;
        goto cleanup;
      }
    }
  }

cleanup:

  return status;
}

DWORD GetArraySize(PEVENT_RECORD pEvent, PTRACE_EVENT_INFO pInfo, USHORT i, PUSHORT ArraySize) {
  DWORD status = ERROR_SUCCESS;
  PROPERTY_DATA_DESCRIPTOR DataDescriptor;
  DWORD PropertySize = 0;

  if ((pInfo->EventPropertyInfoArray[i].Flags & PropertyParamCount) == PropertyParamCount) {
    DWORD Count = 0;  // Expects the count to be defined by a UINT16 or UINT32
    DWORD j = pInfo->EventPropertyInfoArray[i].countPropertyIndex;
    ZeroMemory(&DataDescriptor, sizeof(PROPERTY_DATA_DESCRIPTOR));
    DataDescriptor.PropertyName = (ULONGLONG)((PBYTE)(pInfo) + pInfo->EventPropertyInfoArray[j].NameOffset);
    DataDescriptor.ArrayIndex = ULONG_MAX;
    status = TdhGetPropertySize(pEvent, 0, NULL, 1, &DataDescriptor, &PropertySize);
    status = TdhGetProperty(pEvent, 0, NULL, 1, &DataDescriptor, PropertySize, (PBYTE)&Count);
    *ArraySize = (USHORT)Count;
  } else {
    *ArraySize = pInfo->EventPropertyInfoArray[i].count;
  }

  return status;
}

DWORD GetMapInfo(PEVENT_RECORD pEvent, LPWSTR pMapName, DWORD DecodingSource, PEVENT_MAP_INFO& pMapInfo) {
  DWORD status = ERROR_SUCCESS;
  DWORD MapSize = 0;

  // Retrieve the required buffer size for the map info.

  status = TdhGetEventMapInformation(pEvent, pMapName, pMapInfo, &MapSize);

  if (ERROR_INSUFFICIENT_BUFFER == status) {
    pMapInfo = (PEVENT_MAP_INFO)malloc(MapSize);
    if (pMapInfo == NULL) {
      wprintf(L"Failed to allocate memory for map info (size=%lu).\n", MapSize);
      status = ERROR_OUTOFMEMORY;
      goto cleanup;
    }

    // Retrieve the map info.

    status = TdhGetEventMapInformation(pEvent, pMapName, pMapInfo, &MapSize);
  }

  if (ERROR_SUCCESS == status) {
    if (DecodingSourceXMLFile == DecodingSource) {
      abort();
    }
  } else {
    if (ERROR_NOT_FOUND == status) {
      status = ERROR_SUCCESS;  // This case is okay.
    } else {
      wprintf(L"TdhGetEventMapInformation failed with 0x%x.\n", status);
    }
  }

cleanup:

  return status;
}

LoggingEventRecord LoggingEventRecord::CreateLoggingEventRecord(EVENT_RECORD* pEvent, DWORD& status) {
  LoggingEventRecord ret;
  ret.event_record_ = pEvent;
  status = ERROR_SUCCESS;
  DWORD BufferSize = 0;

  // Retrieve the required buffer size for the event metadata.

  status = TdhGetEventInformation(pEvent, 0, NULL, nullptr, &BufferSize);

  if (ERROR_INSUFFICIENT_BUFFER != status) return ret;
  ret.buffer_.resize(BufferSize);
  // Retrieve the event metadata.
  status = TdhGetEventInformation(pEvent, 0, NULL, ret.GetEventInfo(), &BufferSize);
  return ret;
}

void OrtEventHandler(EVENT_RECORD* pEvent, void* pContext) {
  ProfilingInfo& info = *(ProfilingInfo*)pContext;
  DWORD status = ERROR_SUCCESS;

  LoggingEventRecord record = LoggingEventRecord::CreateLoggingEventRecord(pEvent, status);
  if (ERROR_SUCCESS != status) {
    if (status == ERROR_NOT_FOUND) return;
    wprintf(L"GetEventInformation failed with %lu\n", status);
    abort();
  }
  DWORD PointerSize = 0;
  if (EVENT_HEADER_FLAG_32_BIT_HEADER == (pEvent->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER)) {
    PointerSize = 4;
  } else {
    PointerSize = 8;
  }

  PTRACE_EVENT_INFO pInfo = record.GetEventInfo();
  const wchar_t* name = record.GetTaskName();
  if (wcscmp(name, L"OpEnd") == 0) {
    if (!info.session_started || info.session_ended) return;
    PBYTE pUserData = (PBYTE)pEvent->UserData;
    PBYTE pEndOfUserData = (PBYTE)pEvent->UserData + pEvent->UserDataLength;

    // Print the event data for all the top-level properties. Metadata for all the
    // top-level properties come before structure member properties in the
    // property information array.
    std::wstring opname;
    long time_spent_in_this_op = 0;
    for (USHORT i = 0; i < pInfo->TopLevelPropertyCount; i++) {
      pUserData = PrintProperties(pEvent, pInfo, PointerSize, i, pUserData, pEndOfUserData,
                                  [&opname, &time_spent_in_this_op](const wchar_t* key, wchar_t* value) {
                                    if (wcscmp(key, L"op_name") == 0) {
                                      opname = value;
                                    } else if (wcscmp(key, L"time") == 0) {                                 
                                      time_spent_in_this_op = wcstol(value, nullptr, 10);
                                    } else {
                                      wprintf(key);
                                      abort();
                                    }
                                  });
      if (NULL == pUserData) {
        wprintf(L"Printing top level properties failed.\n");
        abort();
      }
    }
    auto iter = info.op_stat.find(opname);
    if (iter == info.op_stat.end()) {
      OpStat s;
      s.name = opname;
      s.count = 1;
      s.total_time = time_spent_in_this_op;
      info.op_stat[opname] = s;
    } else {
      OpStat& s = iter->second;
      ++s.count;
      s.total_time += time_spent_in_this_op;
    }
  } else if (wcscmp(name, L"OrtRun") == 0) {
    if (!info.session_started || info.session_ended) return;
    if (pInfo->EventDescriptor.Opcode == EVENT_TRACE_TYPE_START) {
      info.op_start_time = pEvent->EventHeader.TimeStamp;
      ++info.ortrun_count;
    } else if (pInfo->EventDescriptor.Opcode == EVENT_TRACE_TYPE_END) {
      if (pEvent->EventHeader.TimeStamp.QuadPart < info.op_start_time.QuadPart) {
        throw std::runtime_error("time error");
      }
      info.time_per_run.push_back(pEvent->EventHeader.TimeStamp.QuadPart - info.op_start_time.QuadPart);
      ++info.ortrun_end_count;
    } else {
      abort();
    }
  }

  else if (wcscmp(name, L"OrtInferenceSessionActivity") == 0) {
    if (pInfo->EventDescriptor.Opcode == EVENT_TRACE_TYPE_START) {
      info.session_started = true;
    } else if (pInfo->EventDescriptor.Opcode == EVENT_TRACE_TYPE_END) {
      info.session_ended = true;
    } else {
      abort();
    }

    printf("OrtInferenceSessionActivity\n");
  } else if (wcscmp(name, L"NodeNameMapping") == 0) {
    // ignore
  } else {
    wprintf(L"unknown event:%s\n", name);
    abort();
  }
}
