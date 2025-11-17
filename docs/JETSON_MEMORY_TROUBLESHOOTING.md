# Jetson NX Memory / NvMap `cuMemHostAlloc` Failures

This document explains the RAM and NvMap issues observed on the Jetson Xavier NX,
why they happen, and what practical steps to take.

It is based on real measurements from this project:

- `Slab` ≈ 10 GB
- `SUnreclaim` ≈ 10 GB
- `kmalloc-128` slab ≈ 9.9 GB
- Errors such as:
  - `NvMapMemAllocInternalTagged: ... error 12`
  - `pipeline error: cuMemHostAlloc failed: out of memory`

## 1. Symptom Summary

- System runs for some time with the ALPR workload active.
- RAM usage (from `tegrastats`) climbs to ~13 GB out of 14.9 GB, swap is in use.
- `lfb` (largest free contiguous block) becomes very small, e.g. `lfb 2x1MB`.
- New ALPR runs (especially OCR with ONNX CUDA) fail with:
  - `NvMapMemAllocInternalTagged: ... error 12`
  - `NvMapMemHandleAlloc: error 0`
  - `pipeline error: cuMemHostAlloc failed: out of memory`
- `dmesg` may not show much, but `/proc/meminfo` and `slabtop` reveal:
  - `Slab` ≈ 10 GB
  - `SUnreclaim` ≈ 10 GB
  - `kmalloc-128` slab ≈ 9.9 GB

In other words, most of the RAM is consumed by **kernel slab allocations** in
`kmalloc-128`, not by user-space processes.

## 2. Root Cause (What Is Happening)

- `kmalloc-128` is a generic kernel cache for small allocations (≤128 bytes).
  Many subsystems use it: GPU drivers, DMA buffers, NvMap, networking, etc.
- In the observed state:
  - `kmalloc-128` ≈ 9.9 GB
  - `SUnreclaim` ≈ 10 GB  
  This means the kernel believes almost 10 GB of small objects are still in use
  and **cannot** be reclaimed.
- Our ALPR workload (oneshot Python runs with ONNXRuntime + CUDA and TensorRT)
  exercises GPU/memory-management code paths heavily. Due to a bug in the
  kernel/driver stack, some of those small allocations are never freed and
  accumulate in `kmalloc-128`.
- Over hours/days and many oneshot ALPR calls, these leaked 128-byte objects pile
  up into gigabytes. Eventually:
  - RAM is almost full.
  - Free space is fragmented (`lfb` is tiny).
  - New CUDA pinned allocations (used by ONNX/TensorRT) fail even though user
    processes do not appear huge in `top`.

This is fundamentally a **kernel/driver leak** in the Jetson stack, triggered by
repeated GPU inference work. It is not a direct bug in the Python code, but the
pattern of repeatedly creating and destroying CUDA/ONNX/TRT contexts makes it
much worse.

## 3. How to Diagnose (Commands)

Run these on the Jetson when the issue appears:

```bash
# High-level memory view
free -h
cat /proc/meminfo | head -n 30

# Slab breakdown (may take a few seconds)
sudo slabtop -sc | head -n 20

# GPU / RAM / lfb snapshot
sudo tegrastats
```

What to look for:

- `/proc/meminfo`:
  - `Slab` large (multi-GB).
  - `SUnreclaim` large (multi-GB), on the same order as `Slab`.
- `slabtop`:
  - `kmalloc-128` showing multi-GB usage (e.g. ~9–10 GB).
- `tegrastats`:
  - `RAM` near the total (e.g. 13/15 GB).
  - `lfb` very small (only a few 1 MB chunks).

If these match, the device is in a state where further CUDA pinned allocations
are likely to fail with NvMap / `cuMemHostAlloc` errors.

## 4. Does Restarting the Jetson Help?

**Yes.** A full reboot resets the kernel slab caches, including `kmalloc-128`.
After a reboot:

- `Slab` and `SUnreclaim` should be small (tens or hundreds of MB, not GB).
- `kmalloc-128` in `slabtop` should be tiny.
- `lfb` in `tegrastats` should be much larger.

However, this is a **temporary** fix:

- As the ALPR workload runs and repeatedly triggers the leaking kernel paths,
  `kmalloc-128` will slowly grow again.
- When it reaches multi-GB size, the same NvMap / `cuMemHostAlloc` failures
  will reappear.

So rebooting is a necessary recovery step, but not a complete solution.

## 5. Immediate Mitigations (Without Changing Architecture)

If you cannot reboot immediately and must keep using the current oneshot PHP
path, you can reduce the pressure on the leaking code paths:

1. **Force OCR to CPU (stop CUDA EP for ONNX)**  
   Edit `php_sendimage.php` and change the environment variables passed to
   `alpr_text_only.py`:

   ```php
   $env = [
       'DET_ENGINE'        => $projectRoot . '/models/detector/yolov9-s_plate_fp16.engine',
       'OCR_ONNX'          => $projectRoot . '/models/ocr/cct_s_v1_global.onnx',
       'PLATE_CONFIG'      => $projectRoot . '/models/ocr/cct_s_v1_global_plate_config.yaml',
       'ONNX_PROVIDER'     => 'cpu',   // <-- was 'cuda'
       'ONNX_GPU_MEM_MB'   => '256',   // safe default if you switch back to CUDA
   ];
   ```

   - This keeps detection on TensorRT (GPU) but runs the OCR ONNX model on CPU.
   - CPU OCR is slower but avoids new CUDA pinned allocations from ONNXRuntime,
     which significantly reduces how fast the leak grows.

2. **Reduce background memory usage where safe**

   - Stop or restart non-essential services:
     - Extra `node` processes, unused containers, heavy background tools.
     - GUI / remote desktop stack if not needed on-site (e.g. gnome-shell, Xorg,
       TeamViewer).
   - This does not shrink the existing `kmalloc-128` leak, but it reduces
     additional pressure and fragmentation.

3. **Monitor slab growth and abort gracefully**

   - Periodically check `slabtop` and `meminfo`. If `kmalloc-128` grows toward
     multi-GB again, expect NvMap errors and plan a controlled restart.

## 6. Recommended Medium-Term Fix (Once You Can Change the Pipeline)

1. **Prefer a warm OCR service over oneshot processes**

   - Use the `alpr-api` service (see `docs/SYSTEMD_API.md`) or a similar long-
     running OCR process on Jetson.
   - Load ONNXRuntime + CUDA and TensorRT **once** at startup, reusing contexts
     for each request instead of creating/destroying them per request.
   - This drastically reduces how often the leaking kernel paths are exercised.

2. **Keep OCR CUDA usage conservative**

   - When using ONNXRuntime with CUDA:
     - Set `ONNX_GPU_MEM_MB` to a modest value (e.g. 256–512 MB) in your
       environment (`ALPR_OCR_ONNX`, `ALPR_PLATE_CONFIG`, `ALPR_ONNX_PROVIDER`,
       `ALPR_ONNX_GPU_MEM_MB`).
   - Avoid unnecessary model reloads or frequent process restarts that re-init
     CUDA and ORT.

3. **Schedule periodic reboots while the leak exists**

   - Until the underlying JetPack/kernel bug is fixed, plan for a **controlled
     reboot window** (e.g. nightly / weekly) to reset `kmalloc-128` and prevent
     reaching the 10 GB slab state during production hours.

4. **Track `kmalloc-128` over time**

   - After reboot, log `slabtop -sc | head -n 20` at regular intervals to a
     file or monitoring system.
   - Correlate big jumps in `kmalloc-128` with ALPR activity (requests per
     minute, model reloads, etc.) so we can further tune the workload.

## 7. When to Escalate / Update JetPack

- If `kmalloc-128` grows rapidly even with:
  - OCR on CPU, and
  - minimal background load,
  it strongly indicates a deeper kernel bug.
- In that case:
  - Check for JetPack / L4T updates that mention memory leaks or NvMap fixes.
  - Consider testing on a separate SD/eMMC image with a newer (or known good)
    JetPack version to confirm whether the leak is version-specific.

## 8. Quick Checklist

When you see `NvMapMemAllocInternalTagged error 12` / `cuMemHostAlloc failed`:

- [ ] Confirm slab leak:
  - `cat /proc/meminfo | head -n 30`
  - `sudo slabtop -sc | head -n 20` → `kmalloc-128` multi-GB?
- [ ] Temporarily force OCR to CPU in `php_sendimage.php` (`ONNX_PROVIDER=cpu`).
- [ ] Stop non-essential memory-heavy services.
- [ ] Plan a controlled reboot to reset `kmalloc-128`.
- [ ] After reboot, monitor `kmalloc-128` growth vs. ALPR activity.

Rebooting **does help** by resetting the kernel’s slab state, but the leak will
recur if the same workload runs for long periods on a kernel/driver stack that
still has the bug. Medium term, the goal is to:

- Reduce how often we stress the leaking paths (warm services instead of
  oneshot), and
- Eventually move to a JetPack / kernel version where `kmalloc-128` does not
  grow uncontrollably under this workload.

