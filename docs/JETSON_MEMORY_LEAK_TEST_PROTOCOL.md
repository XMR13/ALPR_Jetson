# Jetson kmalloc-128 Leak — Test Protocol (ORT CUDA vs Alternatives)

This document defines a reproducible procedure to:

1. Show that the observed multi-GB `kmalloc-128` slab growth is **kernel-side**.
2. Demonstrate that it is **triggered by ONNXRuntime’s CUDAExecutionProvider**
   under our workload.
3. Provide a minimal repro package suitable for NVIDIA support.

It deliberately avoids any kernel rebuild or debug options so it can be run on a
production-like Jetson with only SSH access.

---

## 1. Pre-Checks (Once Per Device)

Run these once so you know the baseline environment.

```bash
uname -a
head -n 1 /etc/nv_tegra_release || cat /etc/os-release

python3 -c "import onnxruntime as ort; print('onnxruntime:', ort.__version__)"
python3 -c "import tensorrt as trt; print('tensorrt:', trt.__version__)"
```

Record:

- Jetson model (Xavier NX, Orin, etc.).
- JetPack / L4T version.
- `onnxruntime-gpu` version.
- TensorRT version.

---

## 2. Test A — App-Level A/B (ONNX CUDA vs CPU)

**Goal:** show that switching the ONNXRuntime provider from `cuda` to `cpu` in
the ALPR API changes the slope of `kmalloc-128` growth, with everything else
held constant.

### 2.1 Setup

1. Ensure `alpr-api` is a systemd **system** service:

   ```bash
   systemctl status alpr-api
   systemctl cat alpr-api | head
   ```

   Confirm:
   - Unit lives under `/etc/systemd/system` or `/lib/systemd/system`.
   - Not under `~/.config/systemd/user`.

2. Confirm the API is enabled at boot:

   ```bash
   systemctl is-enabled alpr-api
   ```

### 2.2 Phase 1 — ONNX CPU Provider

Choose a 12–24 hour window with “normal” traffic (typical PHP/API load).

1. Edit the API env file (example path):

   ```bash
   sudo nano /etc/alpr/alpr-api.env
   ```

   Set:

   ```ini
   ALPR_ONNX_PROVIDER=cpu
   ```

   Do **not** touch detector or other envs.

2. Restart the API:

   ```bash
   sudo systemctl restart alpr-api
   systemctl status alpr-api
   ```

3. Every hour, capture slab + meminfo:

   ```bash
   date
   sudo slabtop -sc | head -n 10
   cat /proc/meminfo | grep -E 'Slab|SUnreclaim'
   ```

   Save this output to a log file (e.g. `cpu_provider_slab.log`).

4. Keep normal ALPR load running (PHP/Apache calling `/v1/alpr` as usual).

Expected outcome:

- `kmalloc-128` either:
  - Remains roughly flat (noise-only changes), or
  - Grows much more slowly than in the problematic runs.

### 2.3 Phase 2 — ONNX CUDA Provider

After Phase 1 completes, repeat the same steps with CUDA:

1. Edit env:

   ```ini
   ALPR_ONNX_PROVIDER=cuda
   ```

2. Restart:

   ```bash
   sudo systemctl restart alpr-api
   ```

3. Same as Phase 1, every hour:

   ```bash
   date
   sudo slabtop -sc | head -n 10
   cat /proc/meminfo | grep -E 'Slab|SUnreclaim'
   ```

   Save to `cuda_provider_slab.log`.

4. Keep ALPR load as similar as possible to Phase 1 (same kind of requests).

Expected outcome:

- If ONNXRuntime CUDA is the trigger:
  - `kmalloc-128` in `cuda_provider_slab.log` will rise significantly over time
    (hundreds of MB to multiple GB).
  - `Slab` and `SUnreclaim` will track that growth.

This A/B test shows, at the **application level**, that changing only the ORT
provider changes `kmalloc-128` behavior.

---

## 3. Test B — Minimal ORT CUDA Repro (No ALPR Code)

**Goal:** isolate ONNXRuntime GPU as the trigger by running a small script that
does nothing but load an ONNX model and call `InferenceSession.run()` in a loop.

This test should be done in a maintenance window, because we stop the ALPR API
to ensure no other GPU user interferes.

### 3.1 Tools Used

We use two helper scripts in `tools/`:

- `tools/ort_kmalloc_probe.py`:
  - Loads an ONNX model.
  - Runs inference in a tight loop with a chosen provider (`cuda` or `cpu`).

- `tools/gpu_slab_watch.sh`:
  - Appends snapshots of `slabtop`, `Slab`, `SUnreclaim`, and `tegrastats` to a
    log file.

### 3.2 Procedure

1. Stop the ALPR API:

   ```bash
   sudo systemctl stop alpr-api
   ```

2. In **Terminal 1**, start slab logging every 5 minutes:

   ```bash
   cd /path/to/ALPR_Jetson
   watch -n 300 "sudo tools/gpu_slab_watch.sh /var/log/ort_cuda_slab.log"
   ```

3. In **Terminal 2**, activate the Python env and run the ORT CUDA probe:

   ```bash
   cd /path/to/ALPR_Jetson
   source venv/bin/activate   # adjust name/path

   python tools/ort_kmalloc_probe.py \
     --onnx models/ocr/cct_s_v2_global.onnx \
     --provider cuda \
     --iters 20000
   ```

   Adjust the `--onnx` path to the actual OCR ONNX model on the Jetson.

4. Let the probe run for 30–60 minutes. `watch` will periodically append slab
   snapshots to `/var/log/ort_cuda_slab.log`.

5. (Optional control) repeat with CPU provider:

   ```bash
   python tools/ort_kmalloc_probe.py \
     --onnx models/ocr/cct_s_v2_global.onnx \
     --provider cpu \
     --iters 20000
   ```

   While continuing to log with `gpu_slab_watch.sh`.

6. Stop `watch`, then restart the API:

   ```bash
   sudo systemctl start alpr-api
   ```

### 3.3 Interpreting Results

Inspect `/var/log/ort_cuda_slab.log`:

- Look at each `==== TIMESTAMP ====` block:
  - Identify the `kmalloc-128` line in the `slabtop` output.
  - Note `Slab` and `SUnreclaim` in `/proc/meminfo`.

If:

- During the `provider=cuda` run, `kmalloc-128` grows steadily and significantly,
  with `SUnreclaim` tracking it, and
- During the `provider=cpu` run, `kmalloc-128` stays almost flat,

then you have a clean repro showing that ONNXRuntime CUDAExecutionProvider,
even **without** the rest of the ALPR code, is sufficient to trigger the slab
leak on this Jetson + JetPack + ORT version combination.

---

## 4. Why We Can’t See “Which Process Allocated kmalloc-128”

On a stock Jetson kernel:

- Slab caches (`kmalloc-128`, `kmalloc-256`, etc.) are **global** kernel
  structures.
- They are not directly accounted to user-space PIDs:
  - `slabtop` has no PID information.
  - `/proc/<pid>/smaps` and `ps aux` only show user-space virtual memory, not
    kernel slab usage.
- Once leaked, slab objects remain in `SUnreclaim` even if the process that
  triggered the allocations exits.

To attach slab allocations to specific code paths, you would normally need:

- A kernel built with debug features (`CONFIG_KMEMLEAK`, `CONFIG_SLUB_DEBUG`),
  and/or
- eBPF/kprobes on `kmem_cache_alloc` / `kmem_cache_free` or cache-specific
  allocators, plus debug symbols.

These are not practical on a production Jetson without custom kernel builds.

Therefore, this protocol relies on **controlled A/B experiments**:

- When only ONNXRuntime CUDA is active, and other GPU workloads are stopped,
  any consistent growth in `kmalloc-128` can be attributed to that path.
- When we flip ORT to CPU or disable it, and the slope collapses, that is strong
  evidence that the ORT CUDA path is the trigger, even though the kernel leak
  itself is in NVIDIA’s code, not ours.

---

## 5. What to Send to NVIDIA

When reporting to NVIDIA, include:

1. Platform details:
   - Output of:

     ```bash
     uname -a
     head -n 1 /etc/nv_tegra_release || cat /etc/os-release
     python3 -c "import onnxruntime as ort; print(ort.__version__)"
     ```

2. The scripts:
   - `tools/ort_kmalloc_probe.py`
   - `tools/gpu_slab_watch.sh`

3. Logs:
   - `cpu_provider_slab.log` (app-level A/B test, ORT provider=cpu).
   - `cuda_provider_slab.log` (app-level A/B test, ORT provider=cuda).
   - `/var/log/ort_cuda_slab.log` from Test B, showing `kmalloc-128` growth
     under the minimal ORT CUDA loop (and ideally the CPU control).

This bundle gives NVIDIA a clear, minimal reproduction of the kernel leak
triggered by ONNXRuntime GPU on your specific Jetson + JetPack environment.

