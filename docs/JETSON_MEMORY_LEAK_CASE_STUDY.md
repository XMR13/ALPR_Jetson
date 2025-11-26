# Jetson NX kmalloc-128 Memory Leak — Case Study & Mitigation Plan

This document summarizes the kmalloc-128 memory leak we observed on the Jetson
Xavier NX, how it manifests in this ALPR project, what we’ve ruled out, and the
mitigations we’re applying until NVIDIA fixes the underlying kernel/driver bug.

It is deliberately thorough so we can hand it to future maintainers or vendors
(e.g., NVIDIA) as a single reference.

---

## 1. Environment & Context

**Hardware**
- Jetson Xavier NX 16 GB

**Software stack**
- JetPack 5.1.5 (L4T for Xavier NX)
- CUDA 11.4
- TensorRT 8.5.2
- Python 3.8.x target
- `onnxruntime-gpu==1.16.3` (from `constraints-jetson.txt`)
- ALPR project from this repo:
  - Detector: YOLOv9 TensorRT engine via `inference/yolov9_trt.py`
  - OCR: ONNXRuntime GPU (`OnnxPlateOCR`) or TensorRT OCR (`OCRService`)
  - FastAPI API server: `src/api_server/server.py`

**Runtime topology (current production)**:
- DeepStream is **not** running.
- The ALPR API runs as a **systemd system service** (`alpr-api.service`) and is
  called synchronously from a PHP/Apache ingress.
- Only a single `uvicorn` worker is used (`--workers 1`) to keep models hot in
  one process.

---

## 2. Observed Symptom

After running the ALPR workload continuously, the Jetson enters a state where:

- `slabtop` shows multi-GB `kmalloc-128` usage:

  ```text
  OBJS ACTIVE  USE OBJ SIZE  SLABS OBJ/SLAB CACHE SIZE NAME
  51689760 51689760 100%    0.12K 1615305       32   6461220K kmalloc-128
  ```

- `/proc/meminfo` shows:
  - `Slab` ≈ several GB
  - `SUnreclaim` ≈ same order as `Slab` (indicating objects the kernel does not
    consider reclaimable)

- `tegrastats` shows:
  - RAM near full,
  - Swap may be in use,
  - `lfb` (largest free contiguous block) very small.

In practice:

- System runs for several days under ALPR load.
- `kmalloc-128` slowly climbs by **~1.5–1.8 GB per day**.
- After ~7–10 days, new GPU allocations fail with:
  - `NvMapMemAllocInternalTagged: ... error 12`
  - `cuMemHostAlloc failed: out of memory`
- Only a **full reboot** resets `Slab` / `SUnreclaim` / `kmalloc-128` back to a
  small baseline and restores normal behavior.

This behavior is summarized and partially documented already in
`docs/JETSON_MEMORY_TROUBLESHOOTING.md`. This case study ties it more precisely
to our current API deployment, ONNXRuntime, and NVIDIA’s kernel/driver behavior.

---

## 3. What Our Program Actually Does (Relevant Parts)

### 3.1 API Process (FastAPI)

The production path is a single FastAPI app:

- `src/api_server/server.py` exposes `/v1/alpr`:
  - Accepts a JPEG image.
  - Runs detector + OCR.
  - Persists events and snapshots (SQLite + JPEG).
  - Returns results as JSON.

Key runtime pieces:

- **Detector (GPU TensorRT)**:
  - Loaded via `inference/yolov9_trt.py`:
    - Uses TensorRT 8.5 and `pycuda.driver` for:
      - `cuda.mem_alloc(...)` (device memory),
      - `cuda.pagelocked_empty(...)` (pinned host memory),
      - A single CUDA stream.
    - Buffers are allocated once per process and reused across inferences.

- **OCR (default: ONNXRuntime GPU)**:
  - When `ALPR_OCR_ONNX` + `ALPR_PLATE_CONFIG` are set:
    - `src/ocr_service/onnx_infer.py::OnnxPlateOCR` constructs an
      `onnxruntime.InferenceSession` with providers:
      - `["CUDAExecutionProvider", "CPUExecutionProvider"]`
      - GPU allocator capped by `ALPR_ONNX_GPU_MEM_MB` (MB).
    - The FastAPI app sets `runtime.ocr_mode = "onnx"` and reuses this single
      `OnnxPlateOCR` instance for all `/v1/alpr` calls.

- **OCR (alternate: TensorRT)**:
  - When `ALPR_OCR_ENGINE` + `ALPR_OCR_CHARSET` are set **and** ONNX env vars
    are not set:
    - `src/ocr_service/trt_infer.py::OCRService` is used instead, loading a TRT
      OCR engine.
    - This also uses `pycuda.driver` buffers, allocated once and reused.

### 3.2 IPC & DeepStream (non-active in current deployment)

The repo contains an optional DS→OCR IPC bridge:

- `src/deepstream_app/crop_probe.cpp` sends JPEG crops over ZeroMQ:
  - Pure CPU path: `cv::imencode` → ZeroMQ PUSH socket → JPEG bytes.
  - No CUDA IPC, no external GPU memory handles.

- `src/ocr_service/app.py` can start a ZeroMQ PULL receiver:
  - Receives JPEGs, decodes via OpenCV, runs OCR via TRT.

**Important**: In the current production system:

- DeepStream is **disabled**.
- The IPC bridge is **not used**.
- There is a **single process** doing both detector and OCR.

---

## 4. Diagnosis: Why kmalloc-128 Grows

### 4.1 What kmalloc-128 Represents

- `kmalloc-128` is a generic kernel slab cache for small allocations (objects
  up to 128 bytes).
- Many subsystems use it:
  - GPU drivers,
  - NvMap / CUDA pinned-memory bookkeeping,
  - DMA descriptors,
  - Networking structures,
  - Filesystem metadata, etc.

When `kmalloc-128` shows multi-GB usage with `SUnreclaim` ~ `Slab`, it means:

- The kernel believes there are gigabytes worth of small objects still in use,
  and it **cannot** reclaim them.
- Even if user-space processes look small in `top` or `ps`, memory is trapped
  in slab caches.

### 4.2 Why the Leak Is Not Our Python/C++ Code

In our code:

- Detector and OCR models are loaded **once** at process start.
- CUDA contexts, streams, and buffers are allocated once and reused.
- The FastAPI server uses a **single worker** to avoid loading models multiple
  times.
- There is no use of:
  - `cudaIpcOpenMemHandle`,
  - `cudaImportExternalMemory`,
  - `cuMemImportFromShareableHandle`,
  - Vulkan external memory APIs,
  - or any other explicit GPU memory IPC.

Despite this:

- `kmalloc-128` steadily grows even when the API process keeps a stable memory
  footprint in user-space.
- The growth persists even after restarting the API process; only a **full
  reboot** shrinks `kmalloc-128`.

This is consistent with a **kernel/driver memory leak**:

- Small kernel objects allocated when handling CUDA / NvMap / pinned memory
  operations are not properly freed.
- Those objects accumulate in `kmalloc-128` as “in use” (`SUnreclaim`), beyond
  the reach of user-space.

### 4.3 Why ONNXRuntime-GPU Is the Primary Trigger

Given the system we’re running:

- With **DeepStream off**, the only GPU workloads are:
  - Detector TensorRT (one context, reused).
  - OCR via ONNXRuntime GPU (CUDAExecutionProvider) when enabled.

ONNXRuntime CUDA provider:

- Uses CUDA APIs (including pinned host memory) under the hood:
  - `cudaMalloc`, `cudaFree`,
  - `cudaHostAlloc` / `cuMemHostAlloc`, etc.
- Those calls exercise NvMap and driver code paths known to interact with
  `kmalloc-128` and other slab caches.

Observationally (from earlier experiments and from the general Jetson
community):

- When ONNXRuntime CUDA is **enabled** and used heavily:
  - `kmalloc-128` grows steadily over hours/days.
- When ONNXRuntime is forced to **CPU only**:
  - `kmalloc-128` growth is dramatically reduced.

This strongly suggests that:

> ONNXRuntime’s CUDAExecutionProvider is the **trigger** for the kernel leak,
> not the location of the bug. The bug lives in NVIDIA’s Jetson driver / L4T
> kernel code that handles GPU allocations and pinned memory.

### 4.4 Why NVIDIA’s External Memory IPC Fix Is Not Our Case

NVIDIA has a documented L4T 35.6.x patch:

- “Fix memory leak observed when importing an external memory handle through
  IPC.”

This fix targets situations where:

- Process A allocates GPU memory and exports a shareable handle.
- Process B imports that handle via CUDA IPC / external memory APIs.

Our deployment:

- Does **not** use CUDA IPC or external memory handles.
- Uses only in-process CUDA via TensorRT and ONNXRuntime.
- Uses ZeroMQ IPC with JPEGs (CPU memory), not GPU memory.

Therefore:

- The external memory IPC fix is not directly relevant to our current
  production leak.
- The leak we see is a **more general driver / NvMap issue** under long-running
  CUDA usage, triggered by ONNXRuntime GPU.

---

## 5. Current Mitigation Strategy

The true root cause is inside NVIDIA’s closed driver / kernel stack for Jetson.
We cannot patch that code ourselves. Instead, we apply mitigations at the
application and operations layers.

### 5.1 App-Level Mitigation: Prefer TensorRT OCR on Jetson

On the Jetson, we treat ONNXRuntime GPU as a **high-risk** path and favor
TensorRT OCR, which:

- Still uses the GPU,
- Avoids ONNXRuntime’s CUDAExecutionProvider,
- Uses a simpler, more Jetson-native path (TensorRT + pycuda) that appears
  less prone to the `kmalloc-128` leak.

Configuration (on Jetson):

- In `/etc/alpr/alpr-api.env` (or equivalent):

  ```ini
  # Detector
  ALPR_DET_ENGINE=/opt/alpr/models/detector/yolov9-s_plate_fp16.engine

  # OCR via TensorRT
  ALPR_OCR_ENGINE=/opt/alpr/models/ocr/ppo_crnn_fp16.engine
  ALPR_OCR_CHARSET=/opt/alpr/models/ocr/charset.txt
  ALPR_OCR_INPUT_WIDTH=160
  ALPR_OCR_INPUT_HEIGHT=32
  ALPR_OCR_CHANNELS=1

  # DO NOT set ALPR_OCR_ONNX / ALPR_PLATE_CONFIG here if you want TRT OCR.
  ```

- Ensure **ONNX env vars are not set**:
  - Do not set `ALPR_OCR_ONNX`, `ALPR_PLATE_CONFIG`, or `ALPR_ONNX_PROVIDER`
    in production. If both ONNX and TRT are configured, the code prefers ONNX.

Result:

- OCR runs through `OCRService` (`src/ocr_service/trt_infer.py`) instead of
  `OnnxPlateOCR`.
- The number of driver/alloc paths exercised by ONNXRuntime CUDA is minimized
  (or eliminated if we keep ONNXGPU off in production).

This does **not** fix the leak in the kernel, but it significantly reduces our
exposure to the specific buggy code path.

### 5.2 Ops-Level Mitigation: Scheduled Reboot

Even with a “safer” GPU path, Jetson kernels have a history of slab leaks under
long-running GPU workloads (DeepStream, CUDA, Vulkan, etc.). To avoid random
OOMs and ensure predictable behavior, we:

- Install a **weekly reboot** service and timer:

  ```ini
  # /etc/systemd/system/weekly-reboot.service
  [Unit]
  Description=Weekly reboot for ALPR Jetson

  [Service]
  Type=oneshot
  ExecStart=/usr/local/sbin/weekly-reboot.sh
  ```

  ```ini
  # /etc/systemd/system/weekly-reboot.timer
  [Unit]
  Description=Trigger weekly reboot

  [Timer]
  OnCalendar=Fri *-*-* 00:00:00
  Persistent=true

  [Install]
  WantedBy=timers.target
  ```

  ```bash
  # /usr/local/sbin/weekly-reboot.sh
  #!/usr/bin/env bash
  logger "Weekly ALPR Jetson reboot in 1 minute"
  wall   "Restart in 1 minute. ALPR API will auto-restart."
  sleep 60
  systemctl reboot
  ```

- Enable the timer:

  ```bash
  sudo systemctl enable weekly-reboot.timer
  sudo systemctl start weekly-reboot.timer
  ```

- Confirm:

  ```bash
  systemctl list-timers | grep weekly-reboot
  ```

Operationally:

- The Jetson reboots in a known low-traffic window (e.g., weekly at 00:00).
- `alpr-api` and Apache/PHP are systemd services with `WantedBy=multi-user.target`,
  so they automatically start on boot **without any user login**.
- Scheduled reboots prevent `kmalloc-128` from reaching multi-GB levels in
  production and collapsing the system at random times.

### 5.3 Monitoring Tools

To monitor the leak and gather evidence for NVIDIA, we added:

1. **Slab/GPU snapshot logger**: `tools/gpu_slab_watch.sh`

   ```bash
   #!/usr/bin/env bash
   # Snapshot slab/mem/GPU state to a log file.

   LOG=${1:-/var/log/gpu_slab_watch.log}
   mkdir -p "$(dirname "$LOG")"

   {
     echo "==== $(date -Iseconds) ===="
     echo "# slabtop"
     slabtop -sc | head -n 15 || echo "slabtop failed"
     echo "# meminfo"
     grep -E 'Slab|SUnreclaim' /proc/meminfo || true
     echo "# tegrastats"
     timeout 3 tegrastats 2>/dev/null | head -n 1 || echo "tegrastats failed"
     echo
   } >> "$LOG"
   ```

2. **ONNXRuntime CUDA probe**: `tools/ort_kmalloc_probe.py`

   - Minimal script that:
     - Loads an ONNX model,
     - Runs `sess.run` in a loop,
     - Lets us observe `kmalloc-128` growth in isolation.

   Usage example (maintenance window):

   ```bash
   # Stop production API so we isolate the probe
   sudo systemctl stop alpr-api

   # Terminal 1: log slab every 5 minutes
   watch -n 300 "sudo tools/gpu_slab_watch.sh /var/log/ort_cuda_slab.log"

   # Terminal 2: hammer ONNXRuntime CUDA
   source venv/bin/activate
   python tools/ort_kmalloc_probe.py \
     --onnx models/ocr/cct_s_v2_global.onnx \
     --provider cuda \
     --iters 20000
   ```

These tools generate a time series that shows `kmalloc-128` and `SUnreclaim`
climbing under ONNXRuntime-GPU load and reset only after reboot.

---

## 6. Future Platform Considerations

### 6.1 Moving to AGX Orin + JetPack 6

It is tempting to assume that upgrading to AGX Orin with JetPack 6.x will
automatically fix the kmalloc-128 leak. Current evidence from the broader
Jetson community suggests:

- JetPack 6.0/6.0 Rev2 had a documented kmalloc-128 leak under DeepStream 7 that
  NVIDIA partially addressed with host1x fence fixes.
- Jetson Linux 36.4.4 added specific patches for memory leaks when importing
  external memory handles through IPC.
- However, as of late 2025, there are still reports of:
  - `kmalloc-128` and `kernfs_node_cache` leaks on Orin NX 16GB with JP 6.1 /
    6.2.1 under various GPU workloads.

Conclusion for planning:

- **Do not assume** “AGX Orin + JetPack 6” automatically eliminates all
  kmalloc-128 leak scenarios.
- Treat Orin as a fresh platform:
  - Run the slab/GPU monitoring tools.
  - Perform soak tests with the ALPR pipeline.
  - Decide on reboot cadence and GPU backends (TRT vs ORT) based on observed
    behavior, not assumptions.

---

## 7. Operational Checklist (Production)

1. **Service wiring**
   - `alpr-api.service` and Apache/PHP are installed under `/etc/systemd/system`
     or distro defaults (not user units).
   - `systemctl is-enabled alpr-api` → `enabled`.
   - `systemctl is-enabled apache2` (or nginx/php-fpm) → `enabled`.

2. **Reboot timer**
   - `weekly-reboot.service` and `weekly-reboot.timer` exist under
     `/etc/systemd/system`.
   - `OnCalendar=...` (capital O) is correct in the timer.
   - `systemctl is-enabled weekly-reboot.timer` → `enabled`.
   - `systemctl list-timers | grep weekly-reboot` shows a valid `NEXT` time.

3. **GPU backend on Jetson**
   - **Production**:
     - Prefer TensorRT OCR:
       - `ALPR_OCR_ENGINE` / `ALPR_OCR_CHARSET` set.
       - `ALPR_OCR_ONNX` / `ALPR_PLATE_CONFIG` **not** set.
   - **Development / experimentation**:
     - ONNXRuntime GPU may be used, but only with slab monitoring and not in
       long-running production mode.

4. **Post-reboot validation**
   - Over SSH, after `sudo reboot`:
     - `uptime` → small “up” time.
     - `systemctl status alpr-api` → `active (running)`.
     - `systemctl status apache2` → `active (running)`.
   - From external machine:
     - `/v1/healthz` and `/v1/alpr` reachable without any local login.

5. **Escalation path**
   - If `kmalloc-128` climbs rapidly even with TensorRT-only OCR:
     - Capture `gpu_slab_watch` logs.
     - Run `ort_kmalloc_probe.py` in a maintenance window.
     - Prepare a minimal repro package (scripts + logs + platform details) to
       send to NVIDIA for driver/kernel investigation.

---

## 8. Summary

- The multi-GB growth of `kmalloc-128` on our Jetson NX under ALPR load is
  caused by a **kernel/driver memory leak** in the NVIDIA Jetson stack,
  triggered by heavy GPU inference workloads (especially ONNXRuntime CUDA).
- Our Python and C++ code do **not** directly leak user-space memory in a way
  that explains the slab behavior; the leak persists after process restarts and
  only resets on reboot.
- We cannot fix the kernel bug ourselves, but we can:
  - Reduce exposure by preferring TensorRT OCR over ONNXRuntime GPU on Jetson.
  - Schedule periodic reboots to keep `kmalloc-128` from reaching
    catastrophic levels.
  - Monitor slab and GPU metrics and provide a minimal repro to NVIDIA.
- Future migrations (e.g., to AGX Orin + JetPack 6) must still treat slab leak
  behavior as an open question and rely on soak tests plus monitoring rather
  than assumptions.

This document, together with `docs/JETSON_MEMORY_TROUBLESHOOTING.md`, should be
kept up to date as we gather more evidence (e.g., from Orin tests or new
JetPack releases) and as mitigations evolve.

