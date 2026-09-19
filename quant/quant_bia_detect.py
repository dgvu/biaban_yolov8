#!/usr/bin/env python3
# Quantize YOLOv8 standard detect model for ZCU104 DPU using Vitis-AI PyTorch quantizer.
# Outputs 3 raw tensors: raw8, raw16, raw32. Each tensor contains box DFL + class logits.

import argparse
import glob
import os
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pytorch_nndct.apis import torch_quantizer

IMG_SIZE = 640


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas


def preprocess_bgr(path, imgsz=640):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Cannot read image: %s" % path)
    img = letterbox(img, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]  # NCHW
    return torch.from_numpy(img).float()


def list_images(image_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(image_dir, e))
    return sorted(files)


def patch_c2f(model):
    cnt = 0

    def c2f_forward(self, x):
        y = self.cv1(x)
        c = self.c if hasattr(self, "c") else y.shape[1] // 2
        y0 = y[:, :c, :, :]
        y1 = y[:, c:, :, :]
        outs = [y0, y1]
        for m in self.m:
            outs.append(m(outs[-1]))
        return self.cv2(torch.cat(outs, 1))

    for m in model.modules():
        if m.__class__.__name__ == "C2f":
            m.forward = types.MethodType(c2f_forward, m)
            cnt += 1
    print("[PATCH] C2f.forward chunk -> slicing:", cnt)


def replace_silu(module):
    cnt = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(0.1, inplace=True))
            cnt += 1
        else:
            cnt += replace_silu(child)
    return cnt


def patch_detect_head(model):
    """
    Patch Detect.forward to return 3 raw combined tensors, one per stride.

    Each output keeps box and class channels concatenated:
        [box_dfl(4 * reg_max), class_logits(nc)]

    For YOLOv8n with reg_max=16 and nc=2:
        C = 4 * 16 + 2 = 66 channels

    Keeping them concatenated avoids strided_slice ops inside the xmodel.
    ARM postprocess will split the channels after DPU inference.
    """
    cnt = 0

    def detect_raw_forward(self, x):
        outs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])   # [N, 4*reg_max, H, W]
            cls = self.cv3[i](x[i])   # [N, nc, H, W]
            raw = torch.cat((box, cls), dim=1)
            outs.append(raw)
        return tuple(outs)

    for m in model.modules():
        if m.__class__.__name__ == "Detect":
            m.forward = types.MethodType(detect_raw_forward, m)
            cnt += 1
    print("[PATCH] Detect.forward -> 3 raw combined outputs, patched heads:", cnt)


class BiaDetectRawWrapper(nn.Module):
    def __init__(self, weight_path):
        super().__init__()
        yolo = YOLO(weight_path)
        self.model = yolo.model.float().eval()
        self.names = yolo.names
        patch_c2f(self.model)
        patch_detect_head(self.model)
        n = replace_silu(self.model)
        print("[PATCH] SiLU -> LeakyReLU:", n, "modules")

    def forward(self, x):
        return self.model(x)


def run_images(qmodel, images, num):
    if num is not None and num > 0:
        images = images[:num]
    print("Images:", len(images))
    with torch.no_grad():
        for i, p in enumerate(images):
            x = preprocess_bgr(p, IMG_SIZE)
            _ = qmodel(x)
            if i % 20 == 0:
                print(f"{i}/{len(images)} {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../runs/detect/bia_yolov8n_std/weights/best.pt")
    ap.add_argument("--image_dir", default="../dataset/train/images")
    ap.add_argument("--quant_mode", choices=["calib", "test"], default="calib")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--output_dir", default="quantize_result")
    args = ap.parse_args()

    images = list_images(args.image_dir)
    if not images:
        raise RuntimeError("No calibration images found in: %s" % args.image_dir)

    model = BiaDetectRawWrapper(args.weights).eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        out = model(dummy)
    print("[CHECK] output type:", type(out), "len=", len(out))
    for i, t in enumerate(out):
        print("  OUT[%d]" % i, tuple(t.shape))

    quantizer = torch_quantizer(args.quant_mode, model, (dummy,), output_dir=args.output_dir)
    qmodel = quantizer.quant_model
    run_images(qmodel, images, args.num)

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print("Exported quant config")
    if args.deploy:
        quantizer.export_xmodel(args.output_dir, deploy_check=False)
        print("Exported xmodel")


if __name__ == "__main__":
    main()
