#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#if __has_include(<opencv2/imgcodecs.hpp>) && __has_include(<opencv2/core.hpp>)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#define ALPR_DS_TOOL_HAS_OPENCV 1
#else
#define ALPR_DS_TOOL_HAS_OPENCV 0
#endif

#include "../src/deepstream_app/crop_probe.hpp"

int main(int argc, char** argv) {
#if !ALPR_DS_TOOL_HAS_OPENCV
    std::cerr << "OpenCV headers not found; rebuild with OpenCV installed" << std::endl;
    return 2;
#else
    if (argc < 2) {
        std::cerr << "Usage: ds_ipc_sender_stub /path/to/crop.jpg" << std::endl;
        return 2;
    }
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cerr << "failed to read image: " << argv[1] << std::endl;
        return 2;
    }
    alpr::ds::CropMetadata meta;
    meta.camera_id = "cam01";
    meta.ts_ms = static_cast<int64_t>(std::time(nullptr)) * 1000;
    meta.frame_id = 1;
    meta.track_id = 1;
    meta.bbox[0] = 0;
    meta.bbox[1] = 0;
    meta.bbox[2] = img.cols;
    meta.bbox[3] = img.rows;
    meta.plate_h = img.rows;
    meta.img_w = img.cols;
    meta.img_h = img.rows;

    const bool ok = alpr::ds::maybe_send_crop_over_ipc(img, meta);
    if (!ok) {
        std::cerr << "maybe_send_crop_over_ipc returned false" << std::endl;
        return 1;
    }
    std::cout << "sent crop to " << alpr::ds::current_config().endpoint << std::endl;
    return 0;
#endif
}
