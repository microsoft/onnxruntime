# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(FS_SOURCES local_filesystem.h sync_api.h controller.h controller.cc)
if(WIN32)
  LIST(APPEND FS_SOURCES local_filesystem_win.cc sync_api_win.cc)
else()
  LIST(APPEND FS_SOURCES local_filesystem_posix.cc sync_api_posix.cc)
endif()
add_library(slim_fs_lib ${FS_SOURCES})
if(WIN32)
  target_compile_definitions(slim_fs_lib PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

if(JPEG_FOUND)
  SET(IMAGE_SRC jpeg_handle.cc jpeg_handle.h jpeg_mem.cc jpeg_mem.h image_loader_libjpeg.cc)
elseif(WIN32)
  SET(IMAGE_SRC image_loader_wic.cc)
endif()
add_executable(image_classifier main.cc runnable_task.h data_processing.h ${IMAGE_SRC}
        async_ring_buffer.h image_loader.cc image_loader.h cached_interpolation.h  single_consumer.h)
if(JPEG_FOUND)
  target_compile_definitions(image_classifier PRIVATE HAVE_JPEG)
  SET(IMAGE_HEADERS ${JPEG_INCLUDE_DIR})
  SET(IMAGE_LIBS ${JPEG_LIBRARIES})
endif()
target_include_directories(image_classifier PRIVATE ${PROJECT_SOURCE_DIR}/include ${IMAGE_HEADERS})
if(WIN32)
  target_compile_definitions(image_classifier PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
target_link_libraries(image_classifier PRIVATE onnxruntime slim_fs_lib ${IMAGE_LIBS})



