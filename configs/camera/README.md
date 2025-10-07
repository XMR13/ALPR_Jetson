Camera config files for RTSP sources.

File naming: one file per camera, e.g., `cam01.rtsp.txt`.

Required keys:
- RTSP_URI: full RTSP URL (include credentials if needed)
- RECONNECT_MAX_RETRIES: `0` for infinite retries
- RECONNECT_INTERVAL_MS: backoff between retries (ms)
- LATENCY_MS: rtspsrc latency buffer (ms)
- DROP_ON_LATENCY: `true` to drop late buffers
- DECODE_CAPS: GStreamer caps for H264/H265

Example: see `cam01.rtsp.txt`.