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

#if ALPR_DS_HAS_ZMQ && ALPR_DS_HAS_OPENCV

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
        hdr << ",\"encoding\":\"jpeg\"";
        hdr << ",\"jpeg_quality\": " << cfg_.jpeg_quality;
        hdr << ",\"priority\": " << meta.priority;
        hdr << "}";

        try {
            zmq::message_t part_hdr(hdr.str());
            zmq::message_t part_jpeg(buf.data(), buf.size());
            const bool ok = socket_->send(part_hdr, zmq::send_flags::sndmore) &&
                            socket_->send(part_jpeg, zmq::send_flags::none);
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

    struct Stats {
        std::atomic<uint64_t> sent{0};
        std::atomic<uint64_t> send_fail{0};
        std::atomic<uint64_t> hwm_drop{0};
        std::atomic<uint64_t> encode_fail{0};
    };

    const Stats& stats() const { return stats_; }

    bool enabled() const { return enabled_.load(); }

private:
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

#endif  // ALPR_DS_HAS_ZMQ && ALPR_DS_HAS_OPENCV

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

}  // namespace ds
}  // namespace alpr
