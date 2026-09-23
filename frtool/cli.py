"""CLI: preprocess | run | postprocess | all"""
from __future__ import annotations

import argparse
import json
import posixpath
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .board import Board
from .config import BOARD, DET, EMB_MODES
from .postprocess import decode_scrfd, load_det_outputs, load_embedding, nms, to_original
from .preprocess import align_face, emb_input_tensor, preprocess_folder


def _session_dir(cfg=BOARD) -> str:
    return posixpath.join(cfg.work_root, time.strftime("%Y%m%d_%H%M%S"))


# ---------------- stage 1 ----------------
def cmd_preprocess(a):
    preprocess_folder(Path(a.images), Path(a.work) / "det_inputs")


# ---------------- stage 2 ----------------
def run_detection(board: Board, work: Path, remote: str, keep: bool) -> Path:
    det_in = work / "det_inputs"
    meta = json.loads((det_in / "meta.json").read_text())
    names = [it["raw"] for it in meta["items"]]
    rdir = posixpath.join(remote, "det")
    board.put_dir(det_in, posixpath.join(rdir, "inputs"))
    out = board.run_model(BOARD.det_model, DET.input_name, names, rdir)
    local_out = work / "det_outputs"
    shutil.rmtree(local_out, ignore_errors=True)
    board.get_tree(out, local_out)
    if not keep:
        board.cleanup(rdir)
    return local_out


def postprocess_detection(work: Path, conf: float, iou: float) -> list[dict]:
    meta = json.loads((work / "det_inputs" / "meta.json").read_text())
    results = []
    for i, it in enumerate(meta["items"]):
        outs = load_det_outputs(work / "det_outputs" / f"Result_{i}")
        d = nms(decode_scrfd(outs, conf), iou)
        d = to_original(d, it["scale"], it["width"], it["height"])
        results.append({"image": it["image"], "width": it["width"], "height": it["height"],
                        "faces": [{"bbox": [round(float(v), 2) for v in r[:4]],
                                   "confidence": round(float(r[4]), 4),
                                   "kps": [[round(float(x), 2), round(float(y), 2)] for x, y in r[5:].reshape(5, 2)]}
                                  for r in d]})
    return results


def preprocess_embedding(work: Path, results: list[dict], mode: str = "arcface") -> list[tuple[int, int, str]]:
    """Align every detected face -> emb_inputs/<img>_f<k>.raw. Returns [(img_idx, face_idx, raw)]."""
    emb = EMB_MODES[mode]
    emb_in = work / "emb_inputs"; shutil.rmtree(emb_in, ignore_errors=True); emb_in.mkdir(parents=True)
    crops = work / "crops"; shutil.rmtree(crops, ignore_errors=True); crops.mkdir()
    items = []
    for ii, r in enumerate(results):
        if not r["faces"]:
            continue
        img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        stem = Path(r["image"]).stem
        for fi, f in enumerate(r["faces"]):
            crop = align_face(img, np.array(f["kps"], np.float32))
            name = f"{stem}_f{fi}.raw"
            emb_input_tensor(crop, emb.layout).tofile(emb_in / name)
            cv2.imwrite(str(crops / f"{stem}_f{fi}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            items.append((ii, fi, name))
    (emb_in / "meta.json").write_text(json.dumps({"mode": mode, "input_name": emb.input_name, "items": items}, indent=2))
    return items


def run_embedding(board: Board, work: Path, remote: str, keep: bool, model: str = BOARD.emb_model) -> Path:
    """model: .so file on the host (uploaded to the session dir), or a file name / absolute path on the board."""
    emb_in = work / "emb_inputs"
    meta = json.loads((emb_in / "meta.json").read_text())
    items = meta["items"]
    local_out = work / "emb_outputs"
    shutil.rmtree(local_out, ignore_errors=True)
    if not items:
        local_out.mkdir(parents=True); return local_out
    rdir = posixpath.join(remote, "emb")
    board.put_dir(emb_in, posixpath.join(rdir, "inputs"))
    if Path(model).is_file():
        remote_model = posixpath.join(rdir, Path(model).name)
        board.sftp.put(str(model), remote_model)
        model = remote_model
    out = board.run_model(model, meta["input_name"], [n for _, _, n in items], rdir)
    board.get_tree(out, local_out)
    if not keep:
        board.cleanup(rdir)
    return local_out


def postprocess_embedding(work: Path, results: list[dict]) -> None:
    meta = json.loads((work / "emb_inputs" / "meta.json").read_text())
    emb = EMB_MODES[meta.get("mode", "arcface")]
    for k, (ii, fi, _) in enumerate(meta["items"]):
        v = load_embedding(work / "emb_outputs" / f"Result_{k}", emb)
        results[ii]["faces"][fi]["embedding"] = [round(float(x), 6) for x in v]


def draw(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        img = cv2.imread(r["image"])
        for f in r["faces"]:
            x1, y1, x2, y2 = map(int, f["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{f['confidence']:.2f}", (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for x, y in f["kps"]:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.imwrite(str(out_dir / Path(r["image"]).name), img)


def _emb_model(a) -> str:
    if a.emb_model:
        return a.emb_model
    if a.emb_mode == "arcface":
        return BOARD.emb_model
    sys.exit(f"--emb-model is required for --emb-mode {a.emb_mode}")


def cmd_run(a):
    work = Path(a.work); remote = _session_dir()
    emb_model = _emb_model(a)
    with Board() as b:
        b.sh(f"mkdir -p {remote}")
        run_detection(b, work, remote, a.keep_remote)
        results = postprocess_detection(work, a.conf, a.iou)
        preprocess_embedding(work, results, a.emb_mode)
        run_embedding(b, work, remote, a.keep_remote, emb_model)
        if not a.keep_remote:
            b.cleanup(remote)
    (work / "detections.json").write_text(json.dumps(results, indent=2))
    print(f"[run] remote session {remote}{' kept' if a.keep_remote else ' cleaned'}; outputs in {work}")


def cmd_postprocess(a):
    work = Path(a.work)
    results = postprocess_detection(work, a.conf, a.iou)
    if (work / "emb_outputs").exists() and (work / "emb_inputs" / "meta.json").exists():
        postprocess_embedding(work, results)
    out = Path(a.output or work / "results.json")
    out.write_text(json.dumps(results, indent=2))
    n_faces = sum(len(r["faces"]) for r in results)
    print(f"[postprocess] {len(results)} images, {n_faces} faces -> {out}")
    if a.draw:
        draw(results, work / "vis"); print(f"[postprocess] visualizations -> {work / 'vis'}")


def cmd_all(a):
    cmd_preprocess(a); cmd_run(a); cmd_postprocess(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fr_tool", description="SCRFD + ArcFace on QRB5165 via qnn-net-run")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, images=False, run=False, post=False):
        sp.add_argument("--work", default="work", help="host working dir (default ./work)")
        if images:
            sp.add_argument("images", help="folder of images")
        if run or post:
            sp.add_argument("--conf", type=float, default=DET.conf_threshold)
            sp.add_argument("--iou", type=float, default=DET.nms_iou)
        if run:
            sp.add_argument("--keep-remote", action="store_true", help="keep /tmp files on the board")
            sp.add_argument("--emb-mode", choices=sorted(EMB_MODES), default="arcface",
                            help="embedding model I/O format (default arcface)")
            sp.add_argument("--emb-model", help="embedding .so: host file (auto-uploaded), or file name in "
                                                "the board model dir / absolute board path "
                                                f"(default for arcface: {BOARD.emb_model})")
        if post:
            sp.add_argument("--output", help="results json path (default <work>/results.json)")
            sp.add_argument("--draw", action="store_true", help="write annotated images to <work>/vis")

    common(sub.add_parser("preprocess", help="images -> det_inputs/*.raw"), images=True)
    common(sub.add_parser("run", help="upload, run both models on board, download outputs"), run=True)
    common(sub.add_parser("postprocess", help="decode det + embeddings -> results.json"), post=True)
    common(sub.add_parser("all", help="preprocess + run + postprocess"), images=True, run=True, post=True)
    a = p.parse_args(argv)
    {"preprocess": cmd_preprocess, "run": cmd_run, "postprocess": cmd_postprocess, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
