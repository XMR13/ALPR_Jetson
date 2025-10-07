cmake_minimum_required(VERSION 3.10)
project(alpr_deepstream CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(alpr_deepstream main.cpp crop_probe.cpp)

# Note: Linking against DeepStream/GStreamer libs requires the Jetson environment.
# These are placeholders; adjust on-device as needed.
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0 gstreamer-video-1.0)

target_include_directories(alpr_deepstream PRIVATE
    ${GSTREAMER_INCLUDE_DIRS}
)

target_link_libraries(alpr_deepstream PRIVATE
    ${GSTREAMER_LIBRARIES}
)

