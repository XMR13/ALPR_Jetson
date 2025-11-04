/**
 * DeepStream ROI crop probe → ZeroMQ IPC sender stub.
 *
 * This file provides a reusable helper for the C++ DeepStream pipeline to
 * deliver plate crops to the Python OCR service over a PUSH socket. It is
 * deliberately self-contained so the stub can compile even when libzmq or
 * OpenCV headers are absent on a developer workstation; the IPC path is only
 * enabled when the required headers are available and the runtime toggle is
 * set (env/config).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "crop_probe.hpp"

#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

#ifdef ALPR_DS_HEADER_HAS_OPENCV
#undef ALPR_DS_HAS_OPENCV
#define ALPR_DS_HAS_OPENCV ALPR_DS_HEADER_HAS_OPENCV
#elif __has_include(<opencv2/opencv.hpp>)
#define ALPR_DS_HAS_OPENCV 1
#else
#define ALPR_DS_HAS_OPENCV 0
#endif

#if __has_include(<zmq.hpp>)
#include <zmq.hpp>
#define ALPR_DS_HAS_ZMQ 1
#else
#define ALPR_DS_HAS_ZMQ 0
#endif

namespace alpr {
namespace ds {

namespace {

inline int _env_to_int(const char* name, int fallback) {
    const char* val = std::getenv(name);
    if (!val) {
        return fallback;
    }
    try {
        return std::stoi(val);
    } catch (...) {
        return fallback;
    }
}

inline bool _env_enabled(const char* name, bool fallback) {
    const char* val = std::getenv(name);
    if (!val) {
        return fallback;
    }
    std::string s(val);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

inline std::string sanitize_json(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '"' || c == '\\') {
            out.push_back('_');
        } else if (static_cast<unsigned char>(c) < 32) {
            out.push_back(' ');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

IpcConfig load_config_from_env() {
    IpcConfig cfg;
    cfg.enabled = _env_enabled("ALPR_DS_IPC_ENABLED", false);
    if (const char* ep = std::getenv("ALPR_DS_IPC_ENDPOINT")) {
        cfg.endpoint = ep;
    }
    cfg.snd_hwm = _env_to_int("ALPR_DS_IPC_SNDHWM", 256);
    cfg.send_timeout_ms = _env_to_int("ALPR_DS_IPC_SEND_TIMEOUT_MS", 10);
    cfg.jpeg_quality = _env_to_int("ALPR_DS_IPC_JPEG_Q", 90);
    cfg.log_send = _env_enabled("ALPR_DS_IPC_LOG", false);
    return cfg;
}

struct ProbeConfig {
    ProbeGating gating;
};

ProbeConfig load_probe_config() {
    ProbeConfig cfg;
    cfg.gating.min_plate_h = _env_to_int("ALPR_DS_IPC_MIN_PLATE_H", 28);
    cfg.gating.priority_only = _env_enabled("ALPR_DS_IPC_PRIORITY_ONLY", false);
    cfg.gating.log_skips = _env_enabled("ALPR_DS_IPC_LOG_SKIPS", false);
    return cfg;
}

struct ProbeCounterState {
    std::atomic<uint64_t> attempted{0};
    std::atomic<uint64_t> skipped_disabled{0};
    std::atomic<uint64_t> skipped_small_h{0};
    std::atomic<uint64_t> skipped_priority{0};
    std::atomic<uint64_t> ipc_sent{0};
    std::atomic<uint64_t> ipc_send_fail{0};
};

ProbeConfig& probe_config_mut() {
    static ProbeConfig cfg = load_probe_config();
    return cfg;
}

ProbeCounterState& probe_counter_state() {
    static ProbeCounterState ctr{};
    return ctr;
}

void maybe_log_skip(const char* reason, const CropMetadata& meta) {
    const auto& gate = probe_config_mut().gating;
    if (!gate.log_skips) {
        return;
    }
    std::cout << "[ds-ipc] skip (" << reason << ") track=" << meta.track_id
              << " frame=" << meta.frame_id << " plate_h=" << meta.plate_h
              << " priority=" << meta.priority << std::endl;
}

#if ALPR_DS_HAS_ZMQ

class IpcSender {
public:
    explicit IpcSender(IpcConfig cfg)
        : cfg_(std::move(cfg)) {
        if (!cfg_.enabled) {
            return;
        }
        try {
            ctx_ = std::make_unique<zmq::context_t>(1);
            socket_ = std::make_unique<zmq::socket_t>(*ctx_, zmq::socket_type::push);
            socket_->setsockopt(ZMQ_SNDHWM, cfg_.snd_hwm);
            socket_->setsockopt(ZMQ_SNDTIMEO, cfg_.send_timeout_ms);
            socket_->connect(cfg_.endpoint);
            enabled_.store(true);
        } catch (const zmq::error_t& err) {
            std::cerr << "[ds-ipc] failed to init socket: " << err.what() << std::endl;
            enabled_.store(false);
        }
    }

#if ALPR_DS_HAS_OPENCV
    bool send(const cv::Mat& bgr, const CropMetadata& meta) {
        if (!enabled_.load()) {
            return false;
        }
        if (!socket_) {
            return false;
        }
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, cfg_.jpeg_quality};
        std::vector<uchar> buf;
        if (!cv::imencode(".jpg", bgr, buf, params)) {
            stats_.encode_fail.fetch_add(1);
            return false;
        }
        return send_buffer(buf.data(), buf.size(), meta, "jpeg", cfg_.jpeg_quality);
    }
#endif

    struct Stats {
        std::atomic<uint64_t> sent{0};
        std::atomic<uint64_t> send_fail{0};
        std::atomic<uint64_t> hwm_drop{0};
        std::atomic<uint64_t> encode_fail{0};
    };

    struct StatsSnapshot {
        uint64_t sent = 0;
        uint64_t send_fail = 0;
        uint64_t hwm_drop = 0;
        uint64_t encode_fail = 0;
    };

    bool send_jpeg(const unsigned char* data, std::size_t size, const CropMetadata& meta) {
        return send_buffer(data, size, meta, "jpeg", cfg_.jpeg_quality);
    }

    StatsSnapshot snapshot() const {
        StatsSnapshot snap;
        snap.sent = stats_.sent.load();
        snap.send_fail = stats_.send_fail.load();
        snap.hwm_drop = stats_.hwm_drop.load();
        snap.encode_fail = stats_.encode_fail.load();
        return snap;
    }

    bool enabled() const { return enabled_.load(); }

private:
    bool send_buffer(const unsigned char* data, std::size_t size, const CropMetadata& meta, const std::string& encoding, int quality) {
        if (!enabled_.load()) {
            return false;
        }
        if (!socket_) {
            return false;
        }
        if (data == nullptr || size == 0) {
            stats_.send_fail.fetch_add(1);
            return false;
        }
        try {
            std::string hdr = make_header(meta, encoding, quality);
            zmq::message_t part_hdr(hdr.data(), hdr.size());
            zmq::message_t part_payload(size);
            std::memcpy(part_payload.data(), data, size);
            const bool ok = socket_->send(part_hdr, zmq::send_flags::sndmore) &&
                            socket_->send(part_payload, zmq::send_flags::none);
            if (ok) {
                stats_.sent.fetch_add(1);
                if (cfg_.log_send) {
                    std::cout << "[ds-ipc] sent crop track=" << meta.track_id
                              << " camera=" << meta.camera_id << std::endl;
                }
            } else {
                stats_.send_fail.fetch_add(1);
            }
            return ok;
        } catch (const zmq::error_t& err) {
            if (err.num() == EAGAIN) {
                stats_.hwm_drop.fetch_add(1);
            } else {
                stats_.send_fail.fetch_add(1);
                std::cerr << "[ds-ipc] send error: " << err.what() << std::endl;
            }
        }
        return false;
    }

    std::string make_header(const CropMetadata& meta, const std::string& encoding, int quality) const {
        std::ostringstream hdr;
        hdr << "{\"version\":1";
        hdr << ",\"camera_id\":\"" << sanitize_json(meta.camera_id) << "\"";
        hdr << ",\"ts_ms\":" << meta.ts_ms;
        hdr << ",\"frame_id\":" << meta.frame_id;
        hdr << ",\"track_id\":" << meta.track_id;
        hdr << ",\"bbox\": [" << meta.bbox[0] << "," << meta.bbox[1] << "," << meta.bbox[2] << "," << meta.bbox[3] << "]";
        hdr << ",\"plate_h\": " << meta.plate_h;
        hdr << ",\"img_w\": " << meta.img_w;
        hdr << ",\"img_h\": " << meta.img_h;
        hdr << ",\"encoding\":\"" << encoding << "\"";
        if (encoding == "jpeg" && quality > 0) {
            hdr << ",\"jpeg_quality\": " << quality;
        }
        hdr << ",\"priority\": " << meta.priority;
        hdr << "}";
        return hdr.str();
    }

    IpcConfig cfg_{};
    std::unique_ptr<zmq::context_t> ctx_;
    std::unique_ptr<zmq::socket_t> socket_;
    std::atomic<bool> enabled_{false};
    Stats stats_{};
};

IpcSender& get_sender() {
    static IpcSender sender(load_config_from_env());
    return sender;
}

#endif  // ALPR_DS_HAS_ZMQ

}  // namespace

bool send_crop_over_ipc(
#if ALPR_DS_HAS_OPENCV
    const cv::Mat& bgr,
#else
    const void* bgr,
#endif
    const CropMetadata& meta) {
#if ALPR_DS_HAS_ZMQ && ALPR_DS_HAS_OPENCV
    if (!get_sender().enabled()) {
        return false;
    }
    return get_sender().send(bgr, meta);
#else
    (void)bgr;
    (void)meta;
    return false;
#endif
}

IpcConfig current_config() {
    return load_config_from_env();
}

CropMetadata make_crop_metadata(const std::string& camera_id,
                                int64_t ts_ms,
                                int frame_id,
                                int track_id,
                                const int bbox[4],
                                int plate_h,
                                int img_w,
                                int img_h,
                                int priority) {
    CropMetadata meta;
    meta.camera_id = camera_id;
    meta.ts_ms = ts_ms;
    meta.frame_id = frame_id;
    meta.track_id = track_id;
    if (bbox) {
        meta.bbox[0] = bbox[0];
        meta.bbox[1] = bbox[1];
        meta.bbox[2] = bbox[2];
        meta.bbox[3] = bbox[3];
    }
    meta.plate_h = plate_h;
    meta.img_w = img_w;
    meta.img_h = img_h;
    meta.priority = priority;
    return meta;
}

bool send_crop_jpeg_over_ipc(const unsigned char* data, std::size_t size, const CropMetadata& meta) {
#if ALPR_DS_HAS_ZMQ
    if (!get_sender().enabled()) {
        return false;
    }
    return get_sender().send_jpeg(data, size, meta);
#else
    (void)data;
    (void)size;
    (void)meta;
    return false;
#endif
}

IpcStats ipc_stats() {
    IpcStats out{};
#if ALPR_DS_HAS_ZMQ
    const auto snap = get_sender().snapshot();
    out.sent = snap.sent;
    out.send_fail = snap.send_fail;
    out.hwm_drop = snap.hwm_drop;
    out.encode_fail = snap.encode_fail;
#endif
    return out;
}

bool ipc_enabled() {
#if ALPR_DS_HAS_ZMQ
    return get_sender().enabled();
#else
    return false;
#endif
}

ProbeGating probe_gating() {
    return probe_config_mut().gating;
}

ProbeCounters probe_counters() {
    ProbeCounters out{};
    const auto& ctr = probe_counter_state();
    out.attempted = ctr.attempted.load();
    out.skipped_disabled = ctr.skipped_disabled.load();
    out.skipped_small_h = ctr.skipped_small_h.load();
    out.skipped_priority = ctr.skipped_priority.load();
    out.ipc_sent = ctr.ipc_sent.load();
    out.ipc_send_fail = ctr.ipc_send_fail.load();
    return out;
}

void reload_probe_gating_from_env() {
    probe_config_mut() = load_probe_config();
}

bool maybe_send_crop_over_ipc(
#if ALPR_DS_HAS_OPENCV
    const cv::Mat& bgr,
#else
    const void* bgr,
#endif
    const CropMetadata& meta) {
    auto& ctr = probe_counter_state();
    ctr.attempted.fetch_add(1);
#if ALPR_DS_HAS_ZMQ && ALPR_DS_HAS_OPENCV
    if (!ipc_enabled()) {
        ctr.skipped_disabled.fetch_add(1);
        maybe_log_skip("ipc_disabled", meta);
        return false;
    }
    const auto gate = probe_config_mut().gating;
    if (gate.priority_only && meta.priority <= 0) {
        ctr.skipped_priority.fetch_add(1);
        maybe_log_skip("priority_only", meta);
        return false;
    }
    if (meta.plate_h > 0 && meta.plate_h < gate.min_plate_h) {
        ctr.skipped_small_h.fetch_add(1);
        maybe_log_skip("min_plate_h", meta);
        return false;
    }
    const bool ok = send_crop_over_ipc(bgr, meta);
    if (ok) {
        ctr.ipc_sent.fetch_add(1);
    } else {
        ctr.ipc_send_fail.fetch_add(1);
    }
    return ok;
#else
    (void)bgr;
    (void)meta;
    ctr.skipped_disabled.fetch_add(1);
    return false;
#endif
}

}  // namespace ds
}  // namespace alpr
