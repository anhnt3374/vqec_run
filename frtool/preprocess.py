"""Host-side preprocessing: images -> float32 .raw files for qnn-net-run."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import DET, EMB, IMAGE_EXTS


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir()
                  if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def letterbox_topleft(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, paste at top-left of a size x size black canvas (RGB).
    Returns (canvas_rgb uint8, scale) where model_coord = orig_coord * scale."""
    h, w = img_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale


def det_input_tensor(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """-> float32 NHWC [1,640,640,3] normalized (x-127.5)/128, and scale."""
    rgb, scale = letterbox_topleft(img_bgr, DET.size)
    x = (rgb.astype(np.float32) - 127.5) / 128.0
    return x[None], scale


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (scale+rotation+translation) mapping src->dst (Umeyama, 5x2)."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    den = (s ** 2).sum()
    if den <= 1e-9:
        raise ValueError("degenerate landmarks")
    a = (s * d).sum() / den
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / den
    M = np.array([[a, -b], [b, a]])
    t = dm - M @ sm
    return np.hstack([M, t[:, None]])  # 2x3, dst = M@src + t


def align_face(img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Warp original image so 5 landmarks land on the ArcFace 112x112 template. Returns RGB uint8."""
    M = similarity_transform(np.asarray(kps, np.float64).reshape(5, 2), np.array(EMB.template))
    crop = cv2.warpAffine(img_bgr, M, (EMB.size, EMB.size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def emb_input_tensor(crop_rgb: np.ndarray) -> np.ndarray:
    """-> float32 NHWC [1,112,112,3] normalized x/127.5-1."""
    return (crop_rgb.astype(np.float32) / 127.5 - 1.0)[None]


def preprocess_folder(image_dir: Path, out_dir: Path) -> dict:
    """Write <stem>.raw per image + meta.json (orig size, scale). Returns meta."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"input_name": DET.input_name, "shape": [1, DET.size, DET.size, 3],
            "dtype": "float32", "items": []}
    for p in list_images(image_dir):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[preprocess] skip unreadable {p}")
            continue
        x, scale = det_input_tensor(img)
        raw = out_dir / f"{p.stem}.raw"
        x.tofile(raw)
        meta["items"].append({"image": str(p.resolve()), "raw": raw.name, "scale": scale,
                              "width": int(img.shape[1]), "height": int(img.shape[0])})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] {len(meta['items'])} images -> {out_dir}")
    return meta
