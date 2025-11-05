#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "crop_probe.hpp"

#if __has_include(<gst/gst.h>)
#define ALPR_HAS_GSTREAMER 1
#include <gst/gst.h>
#include <glib.h>
#else
#define ALPR_HAS_GSTREAMER 0
#endif

#if ALPR_HAS_GSTREAMER && __has_include(<nvds_meta.h>) && __has_include(<nvbufsurface.h>) && __has_include(<nvbufsurftransform.h>)
#define ALPR_HAS_DEEPSTREAM 1
#include <nvbufsurftransform.h>
#include <nvbufsurface.h>
#include <nvds_meta.h>
#else
#define ALPR_HAS_DEEPSTREAM 0
#endif

#if ALPR_HAS_DEEPSTREAM && ALPR_DS_HEADER_HAS_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace {

struct CmdOptions {
    std::string config_path = "configs/deepstream/app_config.txt";
    std::string metadata_log_path;
    bool show_help = false;
};

bool parse_cmd_options(int argc, char** argv, CmdOptions& opts, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
            continue;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) {
                err = "--config requires a value";
                return false;
            }
            opts.config_path = argv[++i];
            continue;
        }
        if (arg.rfind("--config=", 0) == 0) {
            opts.config_path = arg.substr(std::string("--config=").size());
            continue;
        }
        if (arg == "--metadata-log") {
            if (i + 1 >= argc) {
                err = "--metadata-log requires a value";
                return false;
            }
            opts.metadata_log_path = argv[++i];
            continue;
        }
        if (arg.rfind("--metadata-log=", 0) == 0) {
            opts.metadata_log_path = arg.substr(std::string("--metadata-log=").size());
            continue;
        }
        err = "unknown argument: " + arg;
        return false;
    }
    return true;
}

void print_usage() {
    std::cout << "Usage: alpr-deepstream [--config PATH] [--metadata-log PATH]\n";
}

uint64_t wall_clock_ms() {
    const auto now = std::chrono::system_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
}

#if ALPR_HAS_DEEPSTREAM

struct SourceConfig {
    std::string uri;
    std::string camera_id = "source0";
    int type = 4;
    int latency = 200;
    int cudadec_memtype = 0;
};

struct StreamMuxConfig {
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t batch_size = 1;
    uint32_t live_source = 1;
    uint64_t batched_push_timeout = 4000000;  // nanoseconds
};

struct PrimaryGieConfig {
    std::string config_file;
    uint32_t unique_id = 1;
};

struct TrackerConfig {
    bool enabled = false;
    std::string ll_config_file;
    std::string ll_lib_file;
    uint32_t width = 960;
    uint32_t height = 544;
};

struct SinkConfig {
    std::string type = "fakesink";
};

struct AppConfig {
    SourceConfig source;
    StreamMuxConfig streammux;
    PrimaryGieConfig primary;
    TrackerConfig tracker;
    SinkConfig sink;
};

bool load_app_config(const std::string& path, AppConfig& cfg, std::string& err) {
    GError* gerr = nullptr;
    GKeyFile* key_file = g_key_file_new();
    if (!g_key_file_load_from_file(key_file, path.c_str(), G_KEY_FILE_NONE, &gerr)) {
        err = std::string("failed to read config: ") + (gerr ? gerr->message : "unknown error");
        if (gerr) {
            g_error_free(gerr);
        }
        g_key_file_free(key_file);
        return false;
    }

    if (!g_key_file_has_group(key_file, "source0")) {
        err = "config missing [source0] section";
        g_key_file_free(key_file);
        return false;
    }

    GError* local_err = nullptr;
    gchar* uri = g_key_file_get_string(key_file, "source0", "uri", &local_err);
    if (!uri) {
        err = "[source0] uri not set";
        if (local_err) {
            g_error_free(local_err);
        }
        g_key_file_free(key_file);
        return false;
    }
    cfg.source.uri = uri;
    g_free(uri);

    if (g_key_file_has_key(key_file, "source0", "camera-id", nullptr)) {
        gchar* cid = g_key_file_get_string(key_file, "source0", "camera-id", nullptr);
        if (cid) {
            cfg.source.camera_id = cid;
            g_free(cid);
        }
    }
    if (g_key_file_has_key(key_file, "source0", "type", nullptr)) {
        cfg.source.type = g_key_file_get_integer(key_file, "source0", "type", nullptr);
    }
    if (g_key_file_has_key(key_file, "source0", "latency", nullptr)) {
        cfg.source.latency = g_key_file_get_integer(key_file, "source0", "latency", nullptr);
    }
    if (g_key_file_has_key(key_file, "source0", "cudadec-memtype", nullptr)) {
        cfg.source.cudadec_memtype = g_key_file_get_integer(key_file, "source0", "cudadec-memtype", nullptr);
    }

    if (g_key_file_has_group(key_file, "streammux")) {
        if (g_key_file_has_key(key_file, "streammux", "width", nullptr)) {
            cfg.streammux.width = g_key_file_get_integer(key_file, "streammux", "width", nullptr);
        }
        if (g_key_file_has_key(key_file, "streammux", "height", nullptr)) {
            cfg.streammux.height = g_key_file_get_integer(key_file, "streammux", "height", nullptr);
        }
        if (g_key_file_has_key(key_file, "streammux", "batch-size", nullptr)) {
            cfg.streammux.batch_size = g_key_file_get_integer(key_file, "streammux", "batch-size", nullptr);
        }
        if (g_key_file_has_key(key_file, "streammux", "live-source", nullptr)) {
            cfg.streammux.live_source = g_key_file_get_integer(key_file, "streammux", "live-source", nullptr);
        }
        if (g_key_file_has_key(key_file, "streammux", "batched-push-timeout", nullptr)) {
            cfg.streammux.batched_push_timeout = static_cast<uint64_t>(g_key_file_get_integer(key_file, "streammux", "batched-push-timeout", nullptr));
        }
    }

    if (!g_key_file_has_group(key_file, "primary-gie")) {
        err = "config missing [primary-gie] section";
        g_key_file_free(key_file);
        return false;
    }
    gchar* pgie_cfg = g_key_file_get_string(key_file, "primary-gie", "config-file", &local_err);
    if (!pgie_cfg) {
        err = "[primary-gie] config-file not set";
        if (local_err) {
            g_error_free(local_err);
        }
        g_key_file_free(key_file);
        return false;
    }
    cfg.primary.config_file = pgie_cfg;
    g_free(pgie_cfg);
    if (g_key_file_has_key(key_file, "primary-gie", "unique-id", nullptr)) {
        cfg.primary.unique_id = g_key_file_get_integer(key_file, "primary-gie", "unique-id", nullptr);
    }

    if (g_key_file_has_group(key_file, "tracker")) {
        gchar* llcfg = g_key_file_get_string(key_file, "tracker", "ll-config-file", nullptr);
        gchar* lllib = g_key_file_get_string(key_file, "tracker", "ll-lib-file", nullptr);
        if (llcfg && lllib) {
            cfg.tracker.enabled = true;
            cfg.tracker.ll_config_file = llcfg;
            cfg.tracker.ll_lib_file = lllib;
        }
        if (llcfg) {
            g_free(llcfg);
        }
        if (lllib) {
            g_free(lllib);
        }
        if (g_key_file_has_key(key_file, "tracker", "tracker-width", nullptr)) {
            cfg.tracker.width = g_key_file_get_integer(key_file, "tracker", "tracker-width", nullptr);
        }
        if (g_key_file_has_key(key_file, "tracker", "tracker-height", nullptr)) {
            cfg.tracker.height = g_key_file_get_integer(key_file, "tracker", "tracker-height", nullptr);
        }
    }

    if (g_key_file_has_group(key_file, "sink0")) {
        gchar* type = g_key_file_get_string(key_file, "sink0", "type", nullptr);
        if (type) {
            cfg.sink.type = type;
            g_free(type);
        }
    }

    g_key_file_free(key_file);
    return true;
}

struct SourcePadContext {
    GstElement* streammux = nullptr;
    guint index = 0;
};

void on_source_pad_added(GstElement* object, GstPad* pad, gpointer user_data) {
    auto* ctx = static_cast<SourcePadContext*>(user_data);
    if (!ctx || !ctx->streammux) {
        return;
    }
    gchar* pad_name = g_strdup_printf("sink_%u", ctx->index++);
    GstPad* sinkpad = gst_element_get_request_pad(ctx->streammux, pad_name);
    g_free(pad_name);
    if (!sinkpad) {
        std::cerr << "[ds] failed to get streammux sink pad" << std::endl;
        return;
    }
    if (gst_pad_link(pad, sinkpad) != GST_PAD_LINK_OK) {
        std::cerr << "[ds] failed to link source pad to streammux" << std::endl;
    }
    gst_object_unref(sinkpad);
}

#if ALPR_DS_HEADER_HAS_OPENCV
class CropExtractor {
public:
    CropExtractor() = default;
    ~CropExtractor() {
        if (dst_surface_) {
            NvBufSurfaceDestroy(dst_surface_);
        }
    }

    bool ensure_init(int gpu_id) {
        if (gpu_id_ == gpu_id && initialized_) {
            return true;
        }
        NvBufSurfTransformConfigParams config{};
        config.gpu_id = gpu_id;
        config.compute_mode = NvBufSurfTransformCompute_Default;
        config.cuda_stream = nullptr;
        if (NvBufSurfTransformSetSessionParams(&config) != 0) {
            std::cerr << "[ds-ipc] NvBufSurfTransformSetSessionParams failed" << std::endl;
            return false;
        }
        gpu_id_ = gpu_id;
        initialized_ = true;
        return true;
    }

    bool extract(NvBufSurface* batch_surface,
                 guint batch_id,
                 const NvDsObjectMeta* obj_meta,
                 cv::Mat& out_bgr) {
        if (!batch_surface || !obj_meta) {
            return false;
        }
        if (!ensure_init(batch_surface->gpuId)) {
            return false;
        }
        const auto& rect = obj_meta->rect_params;
        const uint32_t width = rect.width > 0 ? static_cast<uint32_t>(rect.width) : 0;
        const uint32_t height = rect.height > 0 ? static_cast<uint32_t>(rect.height) : 0;
        if (width == 0 || height == 0) {
            return false;
        }
        if (!ensure_surface(width, height, batch_surface->gpuId)) {
            return false;
        }

        NvBufSurfTransformRect src_rects[1]{};
        src_rects[0].top = rect.top >= 0 ? static_cast<uint32_t>(rect.top) : 0;
        src_rects[0].left = rect.left >= 0 ? static_cast<uint32_t>(rect.left) : 0;
        src_rects[0].width = width;
        src_rects[0].height = height;

        NvBufSurfTransformRect dst_rects[1]{};
        dst_rects[0].top = 0;
        dst_rects[0].left = 0;
        dst_rects[0].width = width;
        dst_rects[0].height = height;

        NvBufSurfTransformParams transform_params{};
        transform_params.src_rect = src_rects;
        transform_params.dst_rect = dst_rects;
        transform_params.transform_flag = static_cast<NvBufSurfTransform_Transform_Flag>(NVBUFSURF_TRANSFORM_COLOR | NVBUFSURF_TRANSFORM_FILTER);
        transform_params.transform_filter = NvBufSurfTransformInter_Nearest;
        transform_params.transform_flip = NvBufSurfTransform_None;

        NvBufSurface src_surface{};
        src_surface.gpuId = batch_surface->gpuId;
        src_surface.batchSize = 1;
        src_surface.numFilled = 1;
        src_surface.memType = batch_surface->memType;
        src_surface.isContiguous = batch_surface->isContiguous;
        src_surface.surfaceList[0] = batch_surface->surfaceList[batch_id];
        src_surface.planeParams = batch_surface->planeParams;

        const int ret = NvBufSurfTransform(&src_surface, dst_surface_, &transform_params);
        if (ret != 0) {
            std::cerr << "[ds-ipc] NvBufSurfTransform failed: " << ret << std::endl;
            return false;
        }

        if (NvBufSurfaceMap(dst_surface_, 0, 0, NVBUF_MAP_READ) != 0) {
            std::cerr << "[ds-ipc] NvBufSurfaceMap failed" << std::endl;
            return false;
        }
        if (NvBufSurfaceSyncForCpu(dst_surface_, 0, 0) != 0) {
            std::cerr << "[ds-ipc] NvBufSurfaceSyncForCpu failed" << std::endl;
            NvBufSurfaceUnMap(dst_surface_, 0, 0);
            return false;
        }

        auto& params = dst_surface_->surfaceList[0];
        cv::Mat rgba(static_cast<int>(params.height), static_cast<int>(params.width), CV_8UC4,
                     params.mappedAddr.addr[0], params.pitch);
        cv::cvtColor(rgba, out_bgr, cv::COLOR_RGBA2BGR);
        NvBufSurfaceUnMap(dst_surface_, 0, 0);
        return true;
    }

private:
    bool ensure_surface(uint32_t width, uint32_t height, int gpu_id) {
        if (dst_surface_ && width == last_width_ && height == last_height_) {
            return true;
        }
        if (dst_surface_) {
            NvBufSurfaceDestroy(dst_surface_);
            dst_surface_ = nullptr;
        }
        NvBufSurfaceCreateParams create_params{};
        create_params.gpuId = gpu_id;
        create_params.width = width;
        create_params.height = height;
        create_params.layout = NVBUF_LAYOUT_PITCH;
        create_params.memType = NVBUF_MEM_SYSTEM;
        create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        if (NvBufSurfaceCreate(&dst_surface_, 1, &create_params) != 0) {
            std::cerr << "[ds-ipc] NvBufSurfaceCreate failed" << std::endl;
            return false;
        }
        dst_surface_->batchSize = 1;
        dst_surface_->numFilled = 1;
        last_width_ = width;
        last_height_ = height;
        return true;
    }

    bool initialized_ = false;
    int gpu_id_ = 0;
    NvBufSurface* dst_surface_ = nullptr;
    uint32_t last_width_ = 0;
    uint32_t last_height_ = 0;
};
#endif  // ALPR_DS_HEADER_HAS_OPENCV

struct ProbeContext {
    std::vector<std::string> camera_ids;
#if ALPR_DS_HEADER_HAS_OPENCV
    std::unique_ptr<CropExtractor> extractor;
#endif
    bool log_enabled = false;
    std::ofstream metadata_log;
    std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
    std::chrono::milliseconds log_interval{1000};
};

void maybe_log_metrics(ProbeContext* ctx, bool force = false) {
    if (!ctx || !ctx->log_enabled || !ctx->metadata_log.is_open()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!force && now - ctx->last_log < ctx->log_interval) {
        return;
    }
    ctx->last_log = now;
    const auto counters = alpr::ds::probe_counters();
    const auto stats = alpr::ds::ipc_stats();
    ctx->metadata_log << "{\"ts_ms\":" << wall_clock_ms()
                      << ",\"attempted\":" << counters.attempted
                      << ",\"skipped_disabled\":" << counters.skipped_disabled
                      << ",\"skipped_small_h\":" << counters.skipped_small_h
                      << ",\"skipped_priority\":" << counters.skipped_priority
                      << ",\"ipc_sent\":" << counters.ipc_sent
                      << ",\"ipc_send_fail\":" << counters.ipc_send_fail
                      << ",\"ipc_hwm_drop\":" << stats.hwm_drop
                      << ",\"ipc_encode_fail\":" << stats.encode_fail
                      << "}" << std::endl;
    ctx->metadata_log.flush();
}

GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    auto* ctx = static_cast<ProbeContext*>(user_data);
    if (!ctx) {
        return GST_PAD_PROBE_OK;
    }
#if !ALPR_DS_HEADER_HAS_OPENCV
    (void)pad;
    (void)info;
    return GST_PAD_PROBE_OK;
#else
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) {
        return GST_PAD_PROBE_OK;
    }
    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }
    if (!ctx->extractor) {
        ctx->extractor = std::make_unique<CropExtractor>();
    }

    for (NvDsMetaList* frame_list = batch_meta->frame_meta_list; frame_list; frame_list = frame_list->next) {
        auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_list->data);
        if (!frame_meta) {
            continue;
        }
        const guint source_id = frame_meta->source_id;
        std::string camera_id = "cam" + std::to_string(source_id);
        if (source_id < ctx->camera_ids.size()) {
            camera_id = ctx->camera_ids[source_id];
        }
        const int64_t ts_ms = frame_meta->ntp_timestamp ? static_cast<int64_t>(frame_meta->ntp_timestamp / 1000) : static_cast<int64_t>(frame_meta->buf_pts / GST_MSECOND);
        for (NvDsMetaList* obj_list = frame_meta->obj_meta_list; obj_list; obj_list = obj_list->next) {
            auto* obj_meta = static_cast<NvDsObjectMeta*>(obj_list->data);
            if (!obj_meta) {
                continue;
            }
            cv::Mat crop_bgr;
            if (!ctx->extractor->extract(batch_meta->surface, frame_meta->batch_id, obj_meta, crop_bgr)) {
                continue;
            }
            int bbox[4] = {
                static_cast<int>(obj_meta->rect_params.left),
                static_cast<int>(obj_meta->rect_params.top),
                static_cast<int>(obj_meta->rect_params.left + obj_meta->rect_params.width),
                static_cast<int>(obj_meta->rect_params.top + obj_meta->rect_params.height)
            };
            const int plate_h = static_cast<int>(obj_meta->rect_params.height);
            const int priority = (obj_meta->object_id != UNTRACKED_OBJECT_ID && obj_meta->object_id >= 0) ? 1 : 0;
            alpr::ds::CropMetadata meta = alpr::ds::make_crop_metadata(
                camera_id,
                ts_ms,
                static_cast<int>(frame_meta->frame_num),
                static_cast<int>(obj_meta->object_id),
                bbox,
                plate_h,
                static_cast<int>(frame_meta->source_frame_width),
                static_cast<int>(frame_meta->source_frame_height),
                priority);
            alpr::ds::maybe_send_crop_over_ipc(crop_bgr, meta);
        }
    }

    maybe_log_metrics(ctx);
    return GST_PAD_PROBE_OK;
#endif
}

int run_deepstream(const CmdOptions& opts) {
    AppConfig cfg;
    std::string config_err;
    if (!load_app_config(opts.config_path, cfg, config_err)) {
        std::cerr << "[ds] " << config_err << std::endl;
        return 2;
    }
    alpr::ds::reload_probe_gating_from_env();
    const auto gating = alpr::ds::probe_gating();
    const auto ipc_cfg = alpr::ds::current_config();
    std::cout << "[ds-ipc] enabled=" << (ipc_cfg.enabled ? "1" : "0")
              << " endpoint=" << ipc_cfg.endpoint
              << " min_plate_h=" << gating.min_plate_h
              << " priority_only=" << (gating.priority_only ? "1" : "0")
              << std::endl;

    gst_init(nullptr, nullptr);

    GstElement* pipeline = gst_pipeline_new("alpr-deepstream");
    if (!pipeline) {
        std::cerr << "[ds] failed to create pipeline" << std::endl;
        return 1;
    }

    GstElement* source = gst_element_factory_make("nvurisrcbin", "source");
    GstElement* streammux = gst_element_factory_make("nvstreammux", "streammux");
    GstElement* pgie = gst_element_factory_make("nvinfer", "primary-gie");
    GstElement* tracker = nullptr;
    if (cfg.tracker.enabled) {
        tracker = gst_element_factory_make("nvtracker", "tracker");
    }
    GstElement* nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvidconv");
    GstElement* nvosd = gst_element_factory_make("nvdsosd", "nvosd");
    GstElement* sink = gst_element_factory_make(cfg.sink.type.c_str(), "sink");

    if (!source || !streammux || !pgie || !nvvidconv || !nvosd || !sink || (cfg.tracker.enabled && !tracker)) {
        std::cerr << "[ds] failed to create pipeline elements" << std::endl;
        if (source) gst_object_unref(source);
        if (streammux) gst_object_unref(streammux);
        if (pgie) gst_object_unref(pgie);
        if (tracker) gst_object_unref(tracker);
        if (nvvidconv) gst_object_unref(nvvidconv);
        if (nvosd) gst_object_unref(nvosd);
        if (sink) gst_object_unref(sink);
        gst_object_unref(pipeline);
        return 1;
    }

    gst_bin_add_many(GST_BIN(pipeline), source, streammux, pgie, nvvidconv, nvosd, sink, nullptr);
    if (tracker) {
        gst_bin_add(GST_BIN(pipeline), tracker);
    }

    g_object_set(G_OBJECT(source),
                 "uri", cfg.source.uri.c_str(),
                 "latency", cfg.source.latency,
                 "cudadec-memtype", cfg.source.cudadec_memtype,
                 nullptr);

    g_object_set(G_OBJECT(streammux),
                 "batch-size", cfg.streammux.batch_size,
                 "width", cfg.streammux.width,
                 "height", cfg.streammux.height,
                 "live-source", cfg.streammux.live_source,
                 "batched-push-timeout", cfg.streammux.batched_push_timeout,
                 nullptr);

    g_object_set(G_OBJECT(pgie),
                 "config-file-path", cfg.primary.config_file.c_str(),
                 "unique-id", cfg.primary.unique_id,
                 nullptr);

    if (tracker) {
        g_object_set(G_OBJECT(tracker),
                     "ll-config-file", cfg.tracker.ll_config_file.c_str(),
                     "ll-lib-file", cfg.tracker.ll_lib_file.c_str(),
                     "tracker-width", cfg.tracker.width,
                     "tracker-height", cfg.tracker.height,
                     nullptr);
    }

    g_object_set(G_OBJECT(sink), "sync", FALSE, nullptr);

    SourcePadContext src_pad_ctx{streammux, 0};
    g_signal_connect(G_OBJECT(source), "pad-added", G_CALLBACK(on_source_pad_added), &src_pad_ctx);

    bool link_ok = false;
    if (tracker) {
        link_ok = gst_element_link_many(streammux, pgie, tracker, nvvidconv, nvosd, sink, nullptr);
    } else {
        link_ok = gst_element_link_many(streammux, pgie, nvvidconv, nvosd, sink, nullptr);
    }
    if (!link_ok) {
        std::cerr << "[ds] failed to link elements" << std::endl;
        gst_object_unref(pipeline);
        return 1;
    }

    auto ctx_holder = std::make_unique<ProbeContext>();
    ctx_holder->camera_ids.resize(1);
    ctx_holder->camera_ids[0] = cfg.source.camera_id;
    if (!opts.metadata_log_path.empty()) {
        ctx_holder->metadata_log.open(opts.metadata_log_path, std::ios::out | std::ios::app);
        if (!ctx_holder->metadata_log) {
            std::cerr << "[ds] failed to open metadata log: " << opts.metadata_log_path << std::endl;
        } else {
            ctx_holder->log_enabled = true;
        }
    }

    GstPad* osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad) {
        std::cerr << "[ds] failed to get nvosd sink pad" << std::endl;
        gst_object_unref(pipeline);
        return 1;
    }
    gulong probe_id = gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, ctx_holder.get(), nullptr);

    GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    bool running = true;
    while (running) {
        GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                                     static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (!msg) {
            continue;
        }
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_EOS:
                std::cout << "[ds] EOS received" << std::endl;
                running = false;
                break;
            case GST_MESSAGE_ERROR: {
                GError* gerr_msg = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(msg, &gerr_msg, &debug);
                std::cerr << "[ds] error: " << (gerr_msg ? gerr_msg->message : "unknown") << std::endl;
                if (debug) {
                    std::cerr << "[ds] debug: " << debug << std::endl;
                }
                if (gerr_msg) {
                    g_error_free(gerr_msg);
                }
                if (debug) {
                    g_free(debug);
                }
                running = false;
                break;
            }
            default:
                break;
        }
        gst_message_unref(msg);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    if (probe_id != 0) {
        gst_pad_remove_probe(osd_sink_pad, probe_id);
    }
    gst_object_unref(osd_sink_pad);
    gst_object_unref(bus);
    gst_object_unref(pipeline);

    maybe_log_metrics(ctx_holder.get(), true);

    ctx_holder.reset();

    std::cout << "[ds] pipeline stopped" << std::endl;
    return 0;
}

#endif  // ALPR_HAS_DEEPSTREAM

}  // namespace

int main(int argc, char** argv) {
    CmdOptions opts;
    std::string err;
    if (!parse_cmd_options(argc, argv, opts, err)) {
        std::cerr << err << std::endl;
        print_usage();
        return 1;
    }
    if (opts.show_help) {
        print_usage();
        return 0;
    }

#if !ALPR_HAS_GSTREAMER
    std::cerr << "alpr-deepstream built without GStreamer; rebuild on Jetson with DeepStream SDK" << std::endl;
    return 2;
#elif !ALPR_HAS_DEEPSTREAM
    (void)opts;
    std::cerr << "alpr-deepstream built without DeepStream headers; deploy on Jetson to run the full pipeline" << std::endl;
    return 2;
#else
    return run_deepstream(opts);
#endif
}
