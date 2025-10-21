# TensorRT Build Logs

This directory stores raw console outputs from `trtexec` runs so detailed
TensorRT metadata stays out of the high-level docs. For each engine, capture
the Jetson-side build or inspection command and place the log here:

```
docs/build_logs/
  YYYY-MM-DD_trtexec_<engine>.log
```

Suggested commands:

- Build (if creating the engine on Jetson):
  ```
  trtexec --onnx=<model.onnx> --saveEngine=<engine.engine> --fp16 --verbose
  ```
- Inspect an existing engine to recover bindings and formats:
  ```
  trtexec --loadEngine=<engine.engine> --dumpProfile --verbose
  ```

Reference these log files from `docs/OCR_MODEL.md` (or other docs) instead of
embedding the full output inline.
