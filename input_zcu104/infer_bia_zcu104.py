#!/usr/bin/env python3
# Run YOLOv8 detect raw xmodel on ZCU104. Outputs boxes for bullet/target.

import argparse
import json
import os
import re
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
COMBO_BOX_FIRST = True  # xmodel output layout when box+cls are combined: True=[box(64ch),cls(2ch)], False=[cls,box]
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
    combo_c = NC + 4 * REG_MAX
    # NCHW -> NHWC if channel-first with a recognized channel count
    if arr.shape[1] in (4 * REG_MAX, NC, combo_c) and arr.shape[-1] not in (4 * REG_MAX, NC, combo_c):
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
    tmap = {}
    combo_c = NC + 4 * REG_MAX  # combined box+cls channels per stride (66 for NC=2, REG_MAX=16)
    for i, t in enumerate(out_tensors):
        dims = list(t.dims)
        if len(dims) != 4:
            continue
        if dims[1] in (4*REG_MAX, NC, combo_c) and dims[-1] not in (4*REG_MAX, NC, combo_c):
            C, H, W = dims[1], dims[2], dims[3]
        else:
            H, W, C = dims[1], dims[2], dims[3]
        if H not in (80, 40, 20):
            continue
        stride = IMG_SIZE // H
        if C == combo_c:
            tmap[f"out{stride}"] = i
        elif C == 4 * REG_MAX:
            tmap[f"box{stride}"] = i
        elif C == NC:
            tmap[f"cls{stride}"] = i

    combo_ok = all(f"out{s}" in tmap for s in STRIDES)
    split_ok = all(f"box{s}" in tmap and f"cls{s}" in tmap for s in STRIDES)
    if not combo_ok and not split_ok:
        raise RuntimeError(
            f"Missing outputs: need out8/out16/out32 (combined box+cls, {combo_c}ch) "
            f"or box8/box16/box32 + cls8/cls16/cls32 (split), got map={tmap}"
        )
    tmap["_combo"] = combo_ok
    return tmap


def decode(outputs, out_tensors, tmap, img_shape, scale, dw, dh):
    img_h, img_w = img_shape[:2]
    all_boxes = []
    all_scores = []
    all_cls = []
    combo = tmap.get("_combo", False)

    def get(key):
        idx = tmap[key]
        arr = dequant(outputs[idx], tensor_fix(out_tensors[idx]))
        return output_to_nhwc(arr)[0]

    for stride in STRIDES:
        if combo:
            comb = get(f"out{stride}")
            if COMBO_BOX_FIRST:
                box_f, cls_f = comb[..., :4 * REG_MAX], comb[..., 4 * REG_MAX:]
            else:
                cls_f, box_f = comb[..., :NC], comb[..., NC:]
        else:
            box_f = get(f"box{stride}")
            cls_f = get(f"cls{stride}")
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


def infer_image(runner, in_tensors, out_tensors, tmap, img):
    """Run one BGR image through the DPU + host post-process. Returns boxes/scores/cls_ids + timings."""
    inp, scale, dw, dh = preprocess(img, in_tensors[0])
    out_dtype = np.float32 if OUTPUT_FLOAT32_ALREADY_DEQUANT else np.int8
    outputs = [np.zeros(list(t.dims), dtype=out_dtype) for t in out_tensors]
    t0 = time.time()
    job = runner.execute_async([inp], outputs)
    runner.wait(job)
    dpu_ms = (time.time() - t0) * 1000
    t1 = time.time()
    boxes, scores, cls_ids = decode(outputs, out_tensors, tmap, img.shape, scale, dw, dh)
    arm_ms = (time.time() - t1) * 1000
    return boxes, scores, cls_ids, dpu_ms, arm_ms


def natural_key(name):
    # "2.jpg" < "10.jpg" instead of lexicographic "10.jpg" < "2.jpg"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def list_images(folder, ext_csv):
    exts = tuple(e.strip().lower() for e in ext_csv.split(",") if e.strip())
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort(key=natural_key)
    return files


def run_batch(runner, in_tensors, out_tensors, tmap, args):
    files = list_images(args.image, args.ext)
    if not files:
        raise RuntimeError(f"No images with extensions [{args.ext}] found in {args.image}")

    outdir = args.output or "result_images"
    os.makedirs(outdir, exist_ok=True)
    json_path = args.json or "detections.json"

    report = {
        "xmodel": args.xmodel,
        "conf_thresh": CONF_THRESH,
        "iou_thresh": NMS_IOU,
        "input_dir": args.image,
        "output_dir": outdir,
        "images": [],
    }

    print(f"Found {len(files)} images in {args.image}")
    t_start = time.time()
    total_dets = 0
    for n, fname in enumerate(files, 1):
        in_path = os.path.join(args.image, fname)
        out_path = os.path.join(outdir, fname)
        entry = {"file": fname}
        img = cv2.imread(in_path)
        if img is None:
            entry["error"] = "cannot read image"
            report["images"].append(entry)
            print(f"[{n}/{len(files)}] {fname}: cannot read, skipped")
            continue
        try:
            boxes, scores, cls_ids, dpu_ms, arm_ms = infer_image(runner, in_tensors, out_tensors, tmap, img)
        except Exception as e:
            entry["error"] = str(e)
            report["images"].append(entry)
            print(f"[{n}/{len(files)}] {fname}: ERROR {e}")
            continue

        result = draw(img, boxes, scores, cls_ids)
        cv2.imwrite(out_path, result)

        dets = []
        counts = {}
        for box, score, cid in zip(boxes, scores, cls_ids):
            cname = NAMES[int(cid)] if int(cid) < len(NAMES) else str(int(cid))
            counts[cname] = counts.get(cname, 0) + 1
            dets.append({
                "class": cname,
                "class_id": int(cid),
                "confidence": round(float(score), 4),
                "bbox_xyxy": [round(float(v), 2) for v in box],
            })

        entry.update({
            "result_file": out_path,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "num_detections": len(dets),
            "counts": counts,
            "dpu_ms": round(dpu_ms, 2),
            "postprocess_ms": round(arm_ms, 2),
            "detections": dets,
        })
        report["images"].append(entry)
        total_dets += len(dets)
        counts_str = ", ".join(f"{k}={v}" for k, v in counts.items()) or "none"
        print(f"[{n}/{len(files)}] {fname}: {len(dets)} detections ({counts_str}) DPU {dpu_ms:.1f}ms")

    total_s = time.time() - t_start
    report["summary"] = {
        "num_images": len(files),
        "num_ok": sum(1 for im in report["images"] if "error" not in im),
        "num_failed": sum(1 for im in report["images"] if "error" in im),
        "total_detections": total_dets,
        "total_time_s": round(total_s, 2),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(files)} images, {total_dets} detections, {total_s:.1f}s")
    print("Result images ->", outdir)
    print("JSON ->", json_path)


def run(args):
    runner = build_runner(args.xmodel)
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    print("XMODEL:", args.xmodel)
    print("Input tensors:")
    for i,t in enumerate(in_tensors): print(f"  IN[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
    print("Output tensors:")
    for i,t in enumerate(out_tensors): print(f"  OUT[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
    tmap = map_outputs(out_tensors)
    print("Output map:", tmap)

    if os.path.isdir(args.image):
        run_batch(runner, in_tensors, out_tensors, tmap, args)
        return

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError("Cannot read image: " + args.image)
    boxes, scores, cls_ids, dpu_ms, arm_ms = infer_image(runner, in_tensors, out_tensors, tmap, img)
    result = draw(img, boxes, scores, cls_ids)
    out_path = args.output or "result.jpg"
    cv2.imwrite(out_path, result)
    print(f"Detections: {len(boxes)}")
    print(f"DPU: {dpu_ms:.1f} ms")
    print(f"ARM postprocess: {arm_ms:.1f} ms")
    print("Saved ->", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run YOLOv8 detect xmodel on ZCU104. IMAGE can be a single image file, "
                     "or a folder of images to process as a batch."
    )
    ap.add_argument("image", help="Path to one image, or a folder of images (batch mode)")
    ap.add_argument("output", nargs="?", default=None,
                     help="Single mode: output image path (default result.jpg). "
                          "Batch mode: output folder (default result_images)")
    ap.add_argument("--xmodel", default="bia_yolov8_detect.xmodel")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--json", default=None, help="Batch mode: output JSON path (default detections.json)")
    ap.add_argument("--ext", default=".jpg,.jpeg,.png,.bmp",
                     help="Batch mode: comma-separated image extensions to include")
    args = ap.parse_args()
    if args.conf is not None: CONF_THRESH = args.conf
    if args.iou is not None: NMS_IOU = args.iou
    run(args)
