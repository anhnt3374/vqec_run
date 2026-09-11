"""Host-side postprocessing of qnn-net-run float32 outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DET, EMB


def load_det_outputs(result_dir: Path) -> dict[str, np.ndarray]:
    """Read out0..out8 (or semantic-named) .raw files from one Result_N dir."""
    outs = {}
    for i, sem in enumerate(DET.output_names):
        for cand in (f"{sem}.raw", f"out{i}.raw"):
            f = Path(result_dir) / cand
            if f.exists():
                outs[sem] = np.fromfile(f, dtype=np.float32)
                break
        else:
            raise FileNotFoundError(f"missing {sem}/out{i}.raw in {result_dir}")
    return outs


def _sigmoid_if_logit(s: np.ndarray) -> np.ndarray:
    if s.size and (s.max() > 1.0 or s.min() < -1.0):
        return 1.0 / (1.0 + np.exp(-s))
    return s


def decode_scrfd(outs: dict[str, np.ndarray], conf_thr: float = DET.conf_threshold,
                 size: int = DET.size) -> np.ndarray:
    """Returns array [N, 15]: x1,y1,x2,y2,score, 5x(kx,ky) in model (640x640) coordinates."""
    dets = []
    for stride in DET.strides:
        grid = size // stride
        A = DET.anchors_per_cell
        score = _sigmoid_if_logit(outs[f"score_{stride}"].reshape(-1))
        box = outs[f"bbox_{stride}"].reshape(-1, 4)
        kps = outs[f"kps_{stride}"].reshape(-1, 10)
        n = grid * grid * A
        if score.size < n or box.shape[0] < n or kps.shape[0] < n:
            raise ValueError(f"stride {stride}: unexpected output sizes "
                             f"{score.size},{box.shape},{kps.shape} (need {n} anchors)")
        score, box, kps = score[:n], box[:n], kps[:n]
        keep = np.where(np.isfinite(score) & (score >= conf_thr))[0]
        if keep.size == 0:
            continue
        cell = keep // A
        cx = (cell % grid) * stride; cy = (cell // grid) * stride
        b = box[keep] * stride; k = kps[keep] * stride
        x1 = cx - b[:, 0]; y1 = cy - b[:, 1]; x2 = cx + b[:, 2]; y2 = cy + b[:, 3]
        k[:, 0::2] += cx[:, None]; k[:, 1::2] += cy[:, None]
        d = np.column_stack([x1, y1, x2, y2, score[keep], k])
        d[:, [0, 2]] = d[:, [0, 2]].clip(0, size); d[:, [1, 3]] = d[:, [1, 3]].clip(0, size)
        dets.append(d)
    if not dets:
        return np.zeros((0, 15), np.float32)
    d = np.concatenate(dets).astype(np.float32)
    ok = (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1]) & ((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]) >= 100)
    return d[ok]


def nms(dets: np.ndarray, iou_thr: float = DET.nms_iou, max_out: int = DET.max_faces) -> np.ndarray:
    if len(dets) == 0:
        return dets
    order = np.argsort(-dets[:, 4])
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    keep = []
    while order.size and len(keep) < max_out:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(area[i] + area[order[1:]] - inter, 1e-6)
        order = order[1:][iou <= iou_thr]
    return dets[keep]


def to_original(dets: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    """Map model-space detections back to original image pixels (top-left letterbox => divide)."""
    d = dets.copy()
    d[:, [0, 2]] /= scale; d[:, [1, 3]] /= scale; d[:, 5:] /= scale
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, width); d[:, [1, 3]] = d[:, [1, 3]].clip(0, height)
    return d


def load_embedding(result_dir: Path) -> np.ndarray:
    for cand in (f"{EMB.output_name}.raw", "out0.raw"):
        f = Path(result_dir) / cand
        if f.exists():
            v = np.fromfile(f, dtype=np.float32)
            break
    else:
        raise FileNotFoundError(f"no embedding raw in {result_dir}")
    if v.size != EMB.dim:
        raise ValueError(f"embedding has {v.size} values, expected {EMB.dim}")
    if not np.all(np.isfinite(v)):
        raise ValueError("embedding contains non-finite values")
    return v / max(float(np.linalg.norm(v)), 1e-12)
