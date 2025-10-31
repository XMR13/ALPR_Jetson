import argparse, ctypes, glob, os, time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ---- NumPy compatibility shim for TensorRT 8.5 on NumPy >= 1.24 ----
if not hasattr(np, "bool"):
    # TensorRT 8.5 uses np.bool in trt.nptype; define alias to keep it happy
    np.bool = np.bool_


# ----------------- Utilities -----------------
def letterbox(im, new_shape=(640, 640), color=(114,114,114), auto=False, scaleFill=False, scaleup=True):
    """Resize + pad to meet stride-multiple constraints (like YOLO)."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # not used here, keep simple
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right  = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x):
    # x: [..., 4] with [cx, cy, w, h]
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def nms_boxes(boxes, scores, iou_threshold=0.45):
    # boxes: [N,4] xyxy; scores: [N]
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return np.array(keep, dtype=np.int32)

def iou(box, boxes):
    """digunakan untuk meghitung iou antara box dengan boxxes yang lain
    perlu diketahui rumus iou adalah intersection over union (area berpotongan dibagi denagan area gabungan)"""
    # box: [4], boxes: [M,4]
    xx1 = np.maximum(box[0], boxes[:,0]) 
    yy1 = np.maximum(box[1], boxes[:,1])
    xx2 = np.minimum(box[2], boxes[:,2])
    yy2 = np.minimum(box[3], boxes[:,3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / (area1 + area2 - inter + 1e-16)

# ----------------- Plugin Loader -----------------
def _load_trt_plugins(logger):
    """
    Ensure built-in and custom TensorRT plugins are registered before deserializing the engine.
    Optionally set TRT_PLUGIN_PATH to a directory containing custom *.so plugins.
    """
    # 1) NVIDIA built-ins
    try:
        ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        for name in ("libnvinfer_plugin.so.8", "nvinfer_plugin.dll"):
            try:
                ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                pass
    try:
        trt.init_libnvinfer_plugins(logger, "")
    except Exception as e:
        import sys
        print(f"[WARN] init_libnvinfer_plugins() failed: {e}", file=sys.stderr)

    # 2) Custom .so plugins
    plugin_dir = os.environ.get("TRT_PLUGIN_PATH", "")
    if plugin_dir and os.path.isdir(plugin_dir):
        for so in glob.glob(os.path.join(plugin_dir, "*.so")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                import sys
                print(f"[TRT] Loaded custom plugin: {so}", file=sys.stderr)
            except OSError as e:
                import sys
                print(f"[TRT] Failed to load {so}: {e}", file=sys.stderr)

def _print_registered_plugins():
    reg = trt.get_plugin_registry()
    creators = reg.plugin_creator_list
    import sys
    print("[TRT] Registered plugin creators:", file=sys.stderr)
    for c in creators:
        import sys
        print(f" - {c.name} v{c.plugin_version} ns='{c.plugin_namespace}'", file=sys.stderr)

# ---- Safe dtype resolver that avoids trt.nptype (and np.bool) ----
def trt_dtype_to_np(dt):
    # Minimal mapping needed for common YOLO/vision engines
    mapping = {
        trt.DataType.FLOAT:  np.float32,
        trt.DataType.HALF:   np.float16,
        trt.DataType.INT8:   np.int8,
        trt.DataType.INT32:  np.int32,
    }
    # Optional/variant types
    if hasattr(trt.DataType, "BOOL"):
        mapping[trt.DataType.BOOL] = np.bool_  # ✅ no np.bool
    if hasattr(trt.DataType, "UINT8"):
        mapping[trt.DataType.UINT8] = np.uint8
    if dt not in mapping:
        raise TypeError(f"Unsupported TRT dtype: {dt}")
    return mapping[dt]

# ----------------- TensorRT wrapper -----------------
class TRTModule:
    def __init__(self, engine_path, print_plugins=False):
        # Reduce TensorRT log verbosity to keep stdout clean for JSON/text consumers
        logger = trt.Logger(trt.Logger.ERROR)
        _load_trt_plugins(logger)  # ensure plugins are registered

        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                if print_plugins:
                    _print_registered_plugins()
                raise RuntimeError(
                    "Failed to deserialize engine. Missing plugin creators or TRT/CUDA version mismatch."
                )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        self.bindings = [None] * self.engine.num_bindings
        self.binding_addrs = [0] * self.engine.num_bindings
        self.host_buffers = {}
        self.device_buffers = {}
        self.stream = cuda.Stream()

        # collect IO info
        self.input_binding_indices = []
        self.output_binding_indices = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_binding_indices.append(i)
            else:
                self.output_binding_indices.append(i)

        # identify tensor input binding (exclude shape bindings)
        tensor_inputs = [i for i in self.input_binding_indices if not self.engine.is_shape_binding(i)]
        if len(tensor_inputs) != 1:
            raise RuntimeError(
                f"Expected exactly one tensor input, found {len(tensor_inputs)}. "
                "Dynamic batching or multiple inputs are not supported in this helper."
            )
        self.data_input_idx = tensor_inputs[0]
        self._input_hw = None  # type: ignore[assignment]
        self._input_layout = None
        self._input_dtype = trt_dtype_to_np(self.engine.get_binding_dtype(self.data_input_idx))

        if print_plugins:
            _print_registered_plugins()

    def allocate(self, binding_idx, shape):
        dtype = trt_dtype_to_np(self.engine.get_binding_dtype(binding_idx))
        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        self.host_buffers[binding_idx] = (host_mem, dtype, shape)
        self.device_buffers[binding_idx] = device_mem
        self.binding_addrs[binding_idx] = int(device_mem)  # for execute_v2

    def infer(self, inputs_dict):
        # inputs_dict: {binding_idx: np.ndarray}
        # Set dynamic shapes + allocate or reuse
        for idx, arr in inputs_dict.items():
            if self.engine.is_shape_binding(idx):
                self.context.set_shape_input(idx, arr.astype(np.int32))
            else:
                # if dynamic: set binding shape
                if -1 in tuple(self.engine.get_binding_shape(idx)):
                    self.context.set_binding_shape(idx, arr.shape)
                shape = tuple(self.context.get_binding_shape(idx))
                if (idx not in self.host_buffers) or (self.host_buffers[idx][2] != shape):
                    self.allocate(idx, shape)
                np.copyto(self.host_buffers[idx][0].reshape(shape), arr.astype(self.host_buffers[idx][1]))

        # Allocate outputs using shapes resolved by context
        for idx in self.output_binding_indices:
            shape = tuple(self.context.get_binding_shape(idx))
            if (idx not in self.host_buffers) or (self.host_buffers[idx][2] != shape):
                self.allocate(idx, shape)

        # H2D
        for idx in self.input_binding_indices:
            if not self.engine.is_shape_binding(idx):
                host, _, shape = self.host_buffers[idx]
                cuda.memcpy_htod_async(self.device_buffers[idx], host.reshape(shape), self.stream)

        # Execute
        self.context.execute_async_v2(self.binding_addrs, self.stream.handle)

        # D2H outputs
        outputs = {}
        for idx in self.output_binding_indices:
            host, _, shape = self.host_buffers[idx]
            cuda.memcpy_dtoh_async(host.reshape(shape), self.device_buffers[idx], self.stream)
            outputs[idx] = host.reshape(shape)

        self.stream.synchronize()
        return outputs

# ----------------- YOLOv9 postprocess -----------------
def postprocess_yolo(output, img0_shape, input_hw, ratio_pad, conf_thres=0.5, iou_thres=0.45):
    """
    output: [N, 5 + C] with [cx, cy, w, h, obj, cls1..clsC]
    returns: list of (xyxy, score, cls)
    """
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.shape[-1] < 6:
        raise RuntimeError(f"Unexpected output shape {output.shape}; need [N, 5+C].")

    boxes_cxcywh = output[:, :4]
    obj = output[:, 4:5]
    cls_scores = output[:, 5:]
    cls_idx = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
    conf = (obj[:, 0] * cls_conf)

    # filter by conf
    mask = conf > conf_thres
    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    conf = conf[mask]
    cls_idx = cls_idx[mask]

    # from input-space to original image
    xyxy = xywh2xyxy(boxes_cxcywh)
    # If model outputs are normalized [0,1], multiply by input size
    if xyxy.max() <= 2.0:
        xyxy[:, [0,2]] *= input_hw[1]
        xyxy[:, [1,3]] *= input_hw[0]

    # remove padding and scale back
    dw, dh = ratio_pad[1]
    r = ratio_pad[0]
    xyxy[:, [0,2]] -= dw 
    xyxy[:, [1,3]] -= dh 
    xyxy /= r

    # clamp to image
    h0, w0 = img0_shape[:2]
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w0 - 1)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h0 - 1)

    # NMS
    keep = nms_boxes(xyxy, conf, iou_threshold=iou_thres)
    xyxy, conf, cls_idx = xyxy[keep], conf[keep], cls_idx[keep]

    return [(xyxy[i], float(conf[i]), int(cls_idx[i])) for i in range(len(keep))]

def _squeeze01(x):
    # remove a leading batch dim if present
    return x[0] if (x.ndim >= 2 and x.shape[0] == 1) else x

def decode_trt_detections(outputs, img0_shape, input_hw, ratio_pad, conf_thres=0.5, iou_thres=0.45, names=None):
    """
    Supports two formats:
      A) Raw YOLO: single tensor [N, 5+C]  -> use postprocess_yolo()
      B) EfficientNMS: boxes [K,4], scores [K], classes [K] (optionally num_dets)
    Returns: list[(xyxy, score, cls)]
    """
    # --- detect format A: single [N, 5+C] ---
    only = list(outputs.values())
    if len(only) == 1 and only[0].ndim >= 2 and only[0].shape[-1] >= 6:
        out = only[0]
        return postprocess_yolo(
            out, img0_shape, input_hw, ratio_pad, conf_thres=conf_thres, iou_thres=iou_thres
        )

    # --- detect format B: EfficientNMS-style ---
    # Try to identify by shapes
    boxes = scores = classes = num = None
    for t in outputs.values():
        t_sq = _squeeze01(t)
        if t_sq.ndim == 2 and t_sq.shape[-1] == 4:
            boxes = t_sq  # (K,4)
        elif t_sq.ndim == 1:
            # Could be scores or classes or num_dets
            if t_sq.dtype.kind in ("f",):   # float scores
                scores = t_sq
            elif t_sq.dtype.kind in ("i", "u"):  # classes or num
                if t_sq.size == 1:
                    num = int(t_sq.item())
                else:
                    classes = t_sq

    if boxes is None or scores is None or classes is None:
        raise RuntimeError(
            f"Unsupported output set. Got shapes: {[v.shape for v in outputs.values()]}. "
            "Expected either [N,5+C] or (boxes[K,4], scores[K], classes[K] [,num])."
        )

    # Truncate to num_dets if provided
    K = boxes.shape[0]
    if num is not None:
        K = min(K, num)
        boxes = boxes[:K]
        scores = scores[:K]
        classes = classes[:K]

    # Undo letterbox padding/scale: boxes are usually in input image scale (xyxy)
    dw, dh = ratio_pad[1]
    r = ratio_pad[0]
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] -= dw 
    boxes[:, [1, 3]] -= dh 
    boxes /= r

    # Clamp to original image
    h0, w0 = img0_shape[:2]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0 - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0 - 1)

    # Filter by confidence
    m = scores >= conf_thres
    boxes = boxes[m]
    scores = scores[m]
    classes = classes[m]

    # EfficientNMS already applied, so **don’t** run NMS again
    dets = [(boxes[i], float(scores[i]), int(classes[i])) for i in range(boxes.shape[0])]
    return dets

def _resolve_input_spec(trt_model):
    if trt_model._input_hw is None or trt_model._input_layout is None:
        raw_shape = trt_model.engine.get_binding_shape(trt_model.data_input_idx)
        if raw_shape[-1] == 3 and (len(raw_shape) >= 3) and (raw_shape[-2] > 0 or raw_shape[-3] > 0):
            H = raw_shape[-3] if raw_shape[-3] > 0 else 640
            W = raw_shape[-2] if raw_shape[-2] > 0 else 640
            layout = "NHWC"
        else:
            H = raw_shape[-2] if raw_shape[-2] > 0 else 640
            W = raw_shape[-1] if raw_shape[-1] > 0 else 640
            layout = "NCHW"
        trt_model._input_hw = (H, W)
        trt_model._input_layout = layout
    return trt_model._input_hw, trt_model._input_layout

def _prepare_image(trt_model, img0):
    input_hw, layout = _resolve_input_spec(trt_model)
    H, W = input_hw
    img, r, (dw, dh) = letterbox(img0, (H, W), auto=False, scaleFill=False, scaleup=True)
    inp = img.astype(np.float32) / 255.0
    if layout == "NCHW":
        inp = inp.transpose(2, 0, 1)[None, ...]
    else:
        inp = inp[None, ...]
    inp = inp.astype(trt_model._input_dtype, copy=False)
    return trt_model.data_input_idx, inp, input_hw, (r, (dw, dh))

def load_engine(engine_path, print_plugins=False):
    return TRTModule(engine_path, print_plugins=print_plugins)

def infer_image(trt_model, image_path, conf=0.5, iou=0.45):
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise FileNotFoundError(image_path)
    in_idx, inp, input_hw, ratio_pad = _prepare_image(trt_model, img0)
    outputs = trt_model.infer({in_idx: inp})
    dets = decode_trt_detections(
        outputs,
        img0_shape=img0.shape,
        input_hw=input_hw,
        ratio_pad=ratio_pad,
        conf_thres=conf,
        iou_thres=iou,
        names=None,
    )
    return img0, dets

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to TensorRT .engine")
    ap.add_argument("--source", required=True, help="Image path")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS [0..1]")
    ap.add_argument("--labels", default="", help="Optional path to a .txt with class names (one per line)")
    ap.add_argument("--print-plugins", action="store_true", help="Print registered plugin creators")
    args = ap.parse_args()

    if not (0.0 <= args.iou <= 1.0):
        raise ValueError(f"--iou must be in [0,1], got {args.iou}")

    # Load labels if provided
    names = None
    if args.labels and os.path.isfile(args.labels):
        with open(args.labels, "r") as f:
            names = [x.strip() for x in f if x.strip()]

    trt_model = load_engine(args.engine, print_plugins=args.print_plugins)
    img0, dets = infer_image(trt_model, args.source, conf=args.conf, iou=args.iou)
    img_draw = img0.copy()

    # Draw
    for (x1, y1, x2, y2), score, cls in dets:
        p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
        cv2.rectangle(img_draw, p1, p2, (0,255,0), 2)
        label = f"{cls}:{score:.2f}" if names is None or cls >= len(names) else f"{names[cls]}:{score:.2f}"
        cv2.putText(img_draw, label, (p1[0], p1[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    out_path = os.path.splitext(args.source)[0] + f"_pred_{args.conf:.2f}.jpg"
    cv2.imwrite(out_path, img_draw)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
