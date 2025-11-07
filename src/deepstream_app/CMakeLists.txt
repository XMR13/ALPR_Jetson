cmake_minimum_required(VERSION 3.10)
project(alpr_deepstream CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(alpr_deepstream main.cpp crop_probe.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0)
pkg_check_modules(GLIB REQUIRED glib-2.0)

target_include_directories(alpr_deepstream PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
)

target_link_libraries(alpr_deepstream PRIVATE
    ${GSTREAMER_LIBRARIES}
    ${GLIB_LIBRARIES}
)

# Optional OpenCV (crop extraction + JPEG encode)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
    target_link_libraries(alpr_deepstream PRIVATE ${OpenCV_LIBS})
    target_include_directories(alpr_deepstream PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
    message(WARNING "OpenCV not found — build will compile but crop encode path will be disabled")
endif()

# Optional ZeroMQ IPC bridge
find_package(ZeroMQ QUIET)
if(NOT ZeroMQ_FOUND)
    pkg_check_modules(ZMQ QUIET libzmq)
    if(ZMQ_FOUND)
        set(ZeroMQ_LIBRARIES ${ZMQ_LIBRARIES})
        set(ZeroMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIRS})
    endif()
endif()
if(ZeroMQ_LIBRARIES)
    target_link_libraries(alpr_deepstream PRIVATE ${ZeroMQ_LIBRARIES})
    target_include_directories(alpr_deepstream PRIVATE ${ZeroMQ_INCLUDE_DIRS})
else()
    message(WARNING "ZeroMQ not found — build will skip DS→OCR IPC sender")
endif()

# DeepStream SDK (nvbufsurface/nvds_meta/etc.)
set(DEEPSTREAM_SDK_PATH $ENV{DEEPSTREAM_SDK_PATH} CACHE PATH "Path to DeepStream SDK root (/opt/nvidia/deepstream/deepstream)")
if(NOT DEEPSTREAM_SDK_PATH)
    set(DEEPSTREAM_SDK_PATH "/opt/nvidia/deepstream/deepstream")
endif()

set(_ds_include_hints
    "${DEEPSTREAM_SDK_PATH}/sources/includes"
    "${DEEPSTREAM_SDK_PATH}/sources/gst-plugins/gst-nvdsmeta"
    "${DEEPSTREAM_SDK_PATH}/sources/includes/nvdsinfer_custom_impl"
)
find_path(DEEPSTREAM_INCLUDE_DIR nvds_meta.h HINTS ${_ds_include_hints})
if(DEEPSTREAM_INCLUDE_DIR)
    target_include_directories(alpr_deepstream PRIVATE ${DEEPSTREAM_INCLUDE_DIR})
else()
    message(WARNING "DeepStream headers not found — set DEEPSTREAM_SDK_PATH to the SDK root")
endif()

set(_ds_lib_hints
    "${DEEPSTREAM_SDK_PATH}/lib"
    "${DEEPSTREAM_SDK_PATH}/lib/x86_64-linux-gnu"
    "${DEEPSTREAM_SDK_PATH}/lib/aarch64-linux-gnu"
)
find_library(NVBUF_SURFACE_LIB nvbufsurface HINTS ${_ds_lib_hints})
find_library(NVBUF_TRANSFORM_LIB nvbufsurftransform HINTS ${_ds_lib_hints})
find_library(NVDS_META_LIB nvds_meta HINTS ${_ds_lib_hints})

set(_ds_libs)
foreach(lib NVBUF_SURFACE_LIB NVBUF_TRANSFORM_LIB NVDS_META_LIB)
    if(${lib})
        list(APPEND _ds_libs ${${lib}})
    endif()
endforeach()

if(_ds_libs)
    target_link_libraries(alpr_deepstream PRIVATE ${_ds_libs})
else()
    message(WARNING "DeepStream libs not fully resolved — verify DeepStream SDK install when building on Jetson")
endif()
