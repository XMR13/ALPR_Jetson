#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#if __has_include(<opencv2/core.hpp>)
#include <opencv2/core.hpp>
#define ALPR_DS_HEADER_HAS_OPENCV 1
#else
#define ALPR_DS_HEADER_HAS_OPENCV 0
#endif

namespace alpr {
namespace ds {

struct CropMetadata {
    std::string camera_id;
    int64_t ts_ms = 0;
    int frame_id = -1;
    int track_id = -1;
    int bbox[4] = {0, 0, 0, 0};
    int plate_h = 0;
    int img_w = 0;
    int img_h = 0;
    int priority = 0;
};

struct IpcConfig {
    bool enabled = false;
    std::string endpoint = "ipc:///tmp/alpr.ds2ocr.sock";
    int snd_hwm = 256;
    int send_timeout_ms = 10;
    int jpeg_quality = 90;
    bool log_send = false;
};

struct IpcStats {
    uint64_t sent = 0;
    uint64_t send_fail = 0;
    uint64_t hwm_drop = 0;
    uint64_t encode_fail = 0;
};

#if ALPR_DS_HEADER_HAS_OPENCV
bool send_crop_over_ipc(const cv::Mat& bgr, const CropMetadata& meta);
#else
bool send_crop_over_ipc(const void* bgr, const CropMetadata& meta);
#endif

bool send_crop_jpeg_over_ipc(const unsigned char* data, std::size_t size, const CropMetadata& meta);

IpcConfig current_config();
IpcStats ipc_stats();
bool ipc_enabled();

}  // namespace ds
}  // namespace alpr
