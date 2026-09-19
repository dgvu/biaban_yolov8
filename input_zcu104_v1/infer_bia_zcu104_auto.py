#!/usr/bin/env python3
# Run YOLOv8 detect raw xmodel on ZCU104. Supports 6 split outputs or 3 combined outputs.

import argparse
import time
import cv2
import numpy as np
import xir
import vart

IMG_SIZE = 640
NC = 2
REG_MAX = 16
STRIDES = [8, 16, 32]
NAMES = ["bullet", "target"]
CONF_THRESH = 0.25
NMS_IOU = 0.45
MAX_CANDIDATES = 500
OUTPUT_FLOAT32_ALREADY_DEQUANT = False
_GRAPH = None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-9)


def dfl(box_flat):
    n = box_flat.shape[0]
    b = box_flat.reshape(n, 4, REG_MAX)
    p = softmax(b, axis=-1)
    proj = np.arange(REG_MAX, dtype=np.float32)
    return (p * proj).sum(axis=-1)


def tensor_fix(t):
    return int(t.get_attr("fix_point")) if t.has_attr("fix_point") else 0


def dequant(arr, fp):
    x = arr.astype(np.float32)
    if OUTPUT_FLOAT32_ALREADY_DEQUANT:
        return x
    return x / (2 ** fp)


def find_dpu_subgraph(sg):
    if sg.has_attr("device"):
        dev = sg.get_attr("device")
        if isinstance(dev, str) and dev.upper() == "DPU":
            return sg
    for c in sg.toposort_child_subgraph():
        r = find_dpu_subgraph(c)
        if r is not None:
            return r
    return None


def build_runner(xmodel_path):
    global _GRAPH
    _GRAPH = xir.Graph.deserialize(xmodel_path)
    root = _GRAPH.get_root_subgraph()
    dpu = find_dpu_subgraph(root)
    if dpu is None:
        raise RuntimeError("Cannot find DPU subgraph")
    return vart.Runner.create_runner(dpu, "run")


def output_to_nhwc(arr):
    if arr.ndim != 4:
        return arr
    channel_values = (4 * REG_MAX, NC, 4 * REG_MAX + NC)
    # NCHW -> NHWC
    if arr.shape[1] in channel_values and arr.shape[-1] not in channel_values:
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def get_layout_from_dims(dims):
    if len(dims) == 4 and dims[1] == 3:
        return "NCHW"
    return "NHWC"


def letterbox(img_bgr, new_shape=IMG_SIZE, color=(114,114,114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas, r, dw, dh


def preprocess(img_bgr, input_tensor):
    dims = list(input_tensor.dims)
    fp = tensor_fix(input_tensor)
    layout = get_layout_from_dims(dims)
    canvas, scale, dw, dh = letterbox(img_bgr, IMG_SIZE)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":
        x = np.transpose(rgb, (2,0,1))[None]
    else:
        x = rgb[None]
    xq = (x * (2 ** fp)).clip(-128,127).astype(np.int8)
    return np.ascontiguousarray(xq), scale, dw, dh


def map_outputs(out_tensors):
    """Map either:
       - 6 split tensors: box8/cls8, box16/cls16, box32/cls32
       - 3 combined tensors: raw8/raw16/raw32 with 66 channels
    """
    tmap = {}
    total_ch = 4 * REG_MAX + NC

    for i, t in enumerate(out_tensors):
        dims = list(t.dims)
        if len(dims) != 4:
            continue

        channel_values = (4 * REG_MAX, NC, total_ch)
        if dims[1] in channel_values and dims[-1] not in channel_values:
            C, H, W = dims[1], dims[2], dims[3]
        else:
            H, W, C = dims[1], dims[2], dims[3]

        if H not in (80, 40, 20):
            continue

        stride = IMG_SIZE // H
        if C == 4 * REG_MAX:
            tmap[f"box{stride}"] = i
        elif C == NC:
            tmap[f"cls{stride}"] = i
        elif C == total_ch:
            tmap[f"raw{stride}"] = i

    split_ok = all(f"box{s}" in tmap and f"cls{s}" in tmap for s in STRIDES)
    combined_ok = all(f"raw{s}" in tmap for s in STRIDES)

    if split_ok:
        return "split", tmap
    if combined_ok:
        return "combined", tmap

    raise RuntimeError(
        "Cannot map output tensors. Expected either 6 split outputs "
        "(64 box + 2 class) or 3 combined outputs (66 channels). "
        f"Got map={tmap}"
    )


def decode(outputs, out_tensors, mode, tmap, img_shape, scale, dw, dh):
    img_h, img_w = img_shape[:2]
    all_boxes = []
    all_scores = []
    all_cls = []

    def get_tensor(key):
        idx = tmap[key]
        arr = dequant(outputs[idx], tensor_fix(out_tensors[idx]))
        return output_to_nhwc(arr)[0]

    for stride in STRIDES:
        if mode == "split":
            box_f = get_tensor(f"box{stride}")
            cls_f = get_tensor(f"cls{stride}")
        else:
            raw = get_tensor(f"raw{stride}")
            box_f = raw[..., :4 * REG_MAX]
            cls_f = raw[..., 4 * REG_MAX:4 * REG_MAX + NC]
        H, W = box_f.shape[:2]
        cls_scores = sigmoid(cls_f.reshape(-1, NC))
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
        keep = scores > CONF_THRESH
        if keep.sum() == 0:
            continue
        gy, gx = np.mgrid[0:H, 0:W]
        cx = (gx.ravel() + 0.5)[keep] * stride
        cy = (gy.ravel() + 0.5)[keep] * stride
        ltrb = dfl(box_f.reshape(-1, 4 * REG_MAX)[keep]) * stride
        boxes = np.stack([cx - ltrb[:,0], cy - ltrb[:,1], cx + ltrb[:,2], cy + ltrb[:,3]], axis=1)
        s = scores[keep]
        c = cls_ids[keep]
        all_boxes.append(boxes.astype(np.float32))
        all_scores.append(s.astype(np.float32))
        all_cls.append(c.astype(np.int32))

    if not all_boxes:
        return [], [], []
    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    cls_ids = np.concatenate(all_cls, 0)

    # Map from letterbox 640 to original image
    boxes[:, [0,2]] = (boxes[:, [0,2]] - dw) / scale
    boxes[:, [1,3]] = (boxes[:, [1,3]] - dh) / scale
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, img_w - 1)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, img_h - 1)
    valid = ((boxes[:,2] - boxes[:,0]) > 2) & ((boxes[:,3] - boxes[:,1]) > 2)
    boxes, scores, cls_ids = boxes[valid], scores[valid], cls_ids[valid]

    order = np.argsort(-scores)
    if len(order) > MAX_CANDIDATES:
        order = order[:MAX_CANDIDATES]
    boxes, scores, cls_ids = boxes[order], scores[order], cls_ids[order]

    # Class-aware NMS: offset boxes by class
    result_idx = []
    for cid in range(NC):
        inds = np.where(cls_ids == cid)[0]
        if len(inds) == 0:
            continue
        b = boxes[inds]
        s = scores[inds]
        xywh = np.stack([b[:,0], b[:,1], b[:,2]-b[:,0], b[:,3]-b[:,1]], axis=1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), CONF_THRESH, NMS_IOU)
        if len(idx):
            result_idx += inds[np.array(idx).reshape(-1)].tolist()
    result_idx = sorted(result_idx, key=lambda i: -scores[i])
    return boxes[result_idx], scores[result_idx], cls_ids[result_idx]


def draw(img, boxes, scores, cls_ids):
    out = img.copy()
    colors = [(0,255,0), (0,0,255)]
    for box, score, cid in zip(boxes, scores, cls_ids):
        x1,y1,x2,y2 = map(int, box)
        color = colors[int(cid) % len(colors)]
        name = NAMES[int(cid)] if int(cid) < len(NAMES) else str(int(cid))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f"{name} {score:.2f}"
        ty = max(20, y1)
        cv2.rectangle(out, (x1, ty-18), (x1+max(80, len(label)*10), ty+4), color, -1)
        cv2.putText(out, label, (x1+2, ty-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
    return out


def run(args):
    runner = build_runner(args.xmodel)
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    print("XMODEL:", args.xmodel)
    print("Input tensors:")
    for i,t in enumerate(in_tensors): print(f"  IN[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
    print("Output tensors:")
    for i,t in enumerate(out_tensors): print(f"  OUT[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
    mode, tmap = map_outputs(out_tensors)
    print("Output mode:", mode)
    print("Output map:", tmap)
    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Cannot read image: " + args.image)
    inp, scale, dw, dh = preprocess(img, in_tensors[0])
    out_dtype = np.float32 if OUTPUT_FLOAT32_ALREADY_DEQUANT else np.int8
    outputs = [np.zeros(list(t.dims), dtype=out_dtype) for t in out_tensors]
    t0 = time.time()
    job = runner.execute_async([inp], outputs)
    runner.wait(job)
    dpu_ms = (time.time() - t0) * 1000
    t1 = time.time()
    boxes, scores, cls_ids = decode(outputs, out_tensors, mode, tmap, img.shape, scale, dw, dh)
    arm_ms = (time.time() - t1) * 1000
    result = draw(img, boxes, scores, cls_ids)
    cv2.imwrite(args.output, result)
    print(f"Detections: {len(boxes)}")
    print(f"DPU: {dpu_ms:.1f} ms")
    print(f"ARM postprocess: {arm_ms:.1f} ms")
    print("Saved ->", args.output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("output", nargs="?", default="result.jpg")
    ap.add_argument("--xmodel", default="bia_yolov8_detect.xmodel")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    args = ap.parse_args()
    if args.conf is not None: CONF_THRESH = args.conf
    if args.iou is not None: NMS_IOU = args.iou
    run(args)
