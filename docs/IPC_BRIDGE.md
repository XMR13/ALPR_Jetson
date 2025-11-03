# DS→OCR IPC Bridge (Design)

Audience: developer/operator. Scope: local IPC between DeepStream (C++) and OCR service (Python) on Jetson NX (JP 5.1.5, Python 3.8, TRT 8.5.2).

Goals
- Low-latency, low-overhead transfer of plate crops + metadata.
- Back-pressure to avoid OOM; predictable drop policy under load.
- Simple, observable, resilient to transient restarts.

Constraints
- Target: Jetson Xavier NX 16 GB, JetPack 5.1.5, CUDA 11.4, TensorRT 8.5.2.
- Python 3.8 on target; avoid heavy deps. Prefer ZeroMQ over domain sockets.
- DeepStream app is C++; OCR is Python microservice.

Transport Selection
- ZeroMQ over Unix Domain Socket (IPC): `ipc:///tmp/alpr.ds2ocr.sock`
  - Pattern: `PUSH` (DeepStream) → `PULL` (OCR). Simplicity and natural fan-in.
  - Alternative for acks/credit: `DEALER` ↔ `ROUTER` with app-level acks.
  - HWM-based back-pressure using `SNDHWM`/`RCVHWM`.

Message Framing (multipart)
1) Frame 0: UTF-8 JSON header
2) Frame 1: JPEG-encoded crop bytes (or raw BGR when needed; JPEG default)

Header JSON (v1)
{
  "version": 1,
  "camera_id": "cam01",
  "ts_ms": 1730188800000,
  "frame_id": 123456,
  "track_id": 42,
  "bbox": [x1, y1, x2, y2],
  "plate_h": 64,                 // original plate height in px (pre-resize)
  "img_w": 1920,
  "img_h": 1080,
  "encoding": "jpeg",           // jpeg|bgr24
  "jpeg_quality": 90,            // when encoding=jpeg
  "checksum": "sha1:...",        // over payload only
  "priority": 0                  // 0=normal, 1=high (e.g., near center/ROI)
}

Back-Pressure & Drop Policy
- Set `SNDHWM=256` on DS sender; `RCVHWM=256` on OCR receiver.
- If send blocks beyond `send_timeout_ms` (e.g., 10–20 ms), drop lowest-priority or smallest-height plate first.
- Record drops and reasons; expose via `/metrics` (dropped_total, dropped_priority, hwm_block_ms_sum).

Reliability & Recovery
- One-way PUSH→PULL by default. If acks are required later, add a side-channel REQ/REP with `request_id` from header.
- OCR process restarts: PULL socket rebinds to the same path; DS retries `zmq_connect` on EPIPE.
- Clean up stale socket files on service start: remove `/tmp/alpr.ds2ocr.sock` if present.

Security
- IPC path owned by `alpr:alpr`, chmod 660. No external exposure.

Metrics (exported by OCR service)
- `alpr_ipc_rx_total{camera_id}`: messages received
- `alpr_ipc_drop_total{reason}`: drops before OCR (decode fail, hwm, malformed)
- `alpr_ipc_queue_depth`: internal processing queue length
- `alpr_ocr_ms_bucket/sum/count`: OCR latency histogram
- `alpr_end_to_end_ms_bucket/sum/count`: det→ocr combined if `det_ms` included upstream

DeepStream (C++) Sender Sketch (cppzmq)
```cpp
// context and socket
zmq::context_t ctx{1};
zmq::socket_t sock(ctx, zmq::socket_type::push);
sock.setsockopt(ZMQ_SNDHWM, 256);
sock.setsockopt(ZMQ_SNDTIMEO, 10); // ms
sock.connect("ipc:///tmp/alpr.ds2ocr.sock");

// build header JSON (use nlohmann::json)
json hdr = { /* fields from above */ };
std::string hdr_str = hdr.dump();

// encode crop to JPEG (nvjpeg or OpenCV)
std::vector<uchar> jpeg;
cv::imencode(".jpg", crop_bgr, jpeg, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, 90});

// send multipart
zmq::message_t part0(hdr_str.begin(), hdr_str.end());
zmq::message_t part1(jpeg.begin(), jpeg.end());
bool ok = sock.send(part0, zmq::send_flags::sndmore) && sock.send(part1, zmq::send_flags::none);
if (!ok) { /* increment drop metric */ }
```
Helper utilities shipped in `src/deepstream_app/crop_probe.cpp`:
- `send_crop_over_ipc(cv::Mat, CropMetadata)` — encodes to JPEG and publishes (requires OpenCV).
- `send_crop_jpeg_over_ipc(unsigned char*, size_t, CropMetadata)` — publish pre-encoded JPEGs (works even when OpenCV headers are absent).
- `ipc_enabled()` and `ipc_stats()` — quick health snapshot for pad probes and `/metrics`.

Python OCR Receiver Sketch (pyzmq)
```python
import json, zmq, time, hashlib, io, cv2, numpy as np

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PULL)
sock.setsockopt(zmq.RCVHWM, 256)
sock.setsockopt(zmq.RCVTIMEO, 1000)
sock.bind("ipc:///tmp/alpr.ds2ocr.sock")

while True:
    try:
        hdr_b = sock.recv(flags=0)
        payload = sock.recv(flags=0)
    except zmq.Again:
        continue
    try:
        hdr = json.loads(hdr_b.decode("utf-8"))
    except Exception:
        # increment malformed metric
        continue
    if hdr.get("encoding") == "jpeg":
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    else:
        # fallback if raw BGR is used (not default)
        w, h = hdr["w"], hdr["h"]
        img = np.frombuffer(payload, np.uint8).reshape(h, w, 3)
    # push to OCR inference queue; emit result to API
```

Operational Notes
- Socket path: ensure cleanup on service start (remove stale `*.sock`).
- Consider JPEG Q=85–90; evaluate trade-off between latency and OCR quality.
- Add a small bounded in-process queue in OCR with worker threads → batch-friendly `infer_batch`.

Integration Plan
1) Implement OCR-side PULL server in `src/ocr_service/app.py` (toggle via config).
2) Add C++ sender in `src/deepstream_app/crop_probe.cpp` guarded by a compile-time flag. ✅ Stub implemented (ZeroMQ PUSH, env toggles `ALPR_DS_IPC_*`).
3) Expose metrics to `/metrics` and systemd journals.
4) E2E soak test: 1–2 hours; capture queue stats, latency, drop rates.

### Testing the C++ Stub (workstation or Jetson)

1. Build the sender harness (ships in `tools/ds_ipc_sender_stub.cpp`) linking OpenCV + ZeroMQ:

```bash
g++ -std=c++17 -I/usr/include/opencv4 \
    tools/ds_ipc_sender_stub.cpp \
    src/deepstream_app/crop_probe.cpp \
    -lopencv_core -lopencv_imgcodecs -lzmq -o build/ipc_sender_stub
```

2. Run a Python receiver to confirm framing:

```bash
python - <<'PY'
import json, zmq
ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PULL)
sock.bind('ipc:///tmp/alpr.ds2ocr.sock')
hdr = json.loads(sock.recv())
payload = sock.recv()
print('header:', hdr)
print('payload-bytes:', len(payload))
PY
```

3. From another shell, export runtime vars and send a dummy crop:

```bash
export ALPR_DS_IPC_ENABLED=1
export ALPR_DS_IPC_ENDPOINT=ipc:///tmp/alpr.ds2ocr.sock
export ALPR_DS_IPC_LOG=1
./build/ipc_sender_stub /path/to/crop.jpg
```

4. Verify receiver prints the header (`encoding=jpeg`, `camera_id`, etc.) and payload length.

At runtime, `send_crop_over_ipc` can be invoked from the DeepStream probe with a `cv::Mat` crop and `CropMetadata` (camera, track, bbox). When libzmq or OpenCV is unavailable, the helper no-ops safely.


Config Snippet (YAML)
```yaml
ipc:
  enabled: true
  endpoint: "ipc:///tmp/alpr.ds2ocr.sock"
  jpeg_quality: 90
  snd_hwm: 256
  rcv_hwm: 256
  send_timeout_ms: 10
  recv_timeout_ms: 1000
  workers: 2
  batch_size: 8
```
