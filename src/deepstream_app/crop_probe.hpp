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

struct ProbeGating {
    int min_plate_h = 28;
    bool priority_only = false;
    bool log_skips = false;
};

struct ProbeCounters {
    uint64_t attempted = 0;
    uint64_t skipped_disabled = 0;
    uint64_t skipped_small_h = 0;
    uint64_t skipped_priority = 0;
    uint64_t ipc_sent = 0;
    uint64_t ipc_send_fail = 0;
};

CropMetadata make_crop_metadata(const std::string& camera_id,
                                int64_t ts_ms,
                                int frame_id,
                                int track_id,
                                const int bbox[4],
                                int plate_h,
                                int img_w,
                                int img_h,
                                int priority);

#if ALPR_DS_HEADER_HAS_OPENCV
bool send_crop_over_ipc(const cv::Mat& bgr, const CropMetadata& meta);
#else
bool send_crop_over_ipc(const void* bgr, const CropMetadata& meta);
#endif

bool send_crop_jpeg_over_ipc(const unsigned char* data, std::size_t size, const CropMetadata& meta);

IpcConfig current_config();
IpcStats ipc_stats();
bool ipc_enabled();

ProbeGating probe_gating();
ProbeCounters probe_counters();
void reload_probe_gating_from_env();

#if ALPR_DS_HEADER_HAS_OPENCV
bool maybe_send_crop_over_ipc(const cv::Mat& bgr, const CropMetadata& meta);
#else
bool maybe_send_crop_over_ipc(const void* bgr, const CropMetadata& meta);
#endif

}  // namespace ds
}  // namespace alpr
