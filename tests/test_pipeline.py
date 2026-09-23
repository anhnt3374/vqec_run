import numpy as np, cv2, json
from pathlib import Path
from frtool.preprocess import det_input_tensor, similarity_transform, align_face, emb_input_tensor, preprocess_folder
from frtool.postprocess import decode_scrfd, nms, to_original, load_det_outputs
from frtool.config import DET, EMB


def test_det_input_tensor_letterbox_topleft():
    img = np.full((480, 640, 3), 255, np.uint8)
    x, scale = det_input_tensor(img)
    assert x.shape == (1, 640, 640, 3) and x.dtype == np.float32 and scale == 1.0
    assert np.isclose(x[0, 0, 0, 0], (255 - 127.5) / 128) and np.isclose(x[0, 479, 639, 0], (255 - 127.5) / 128)
    assert np.isclose(x[0, 480, 0, 0], -127.5 / 128)  # padding is black


def test_similarity_transform_maps_template_to_itself():
    t = np.array(EMB.template)
    M = similarity_transform(t, t)
    assert np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-9)
    # scale 2 + shift
    M = similarity_transform(t * 2 + [10, 5], t)
    src = np.hstack([t * 2 + [10, 5], np.ones((5, 1))])
    assert np.allclose(src @ M.T, t, atol=1e-6)


def test_align_and_emb_tensor():
    img = np.random.randint(0, 255, (300, 300, 3), np.uint8)
    crop = align_face(img, np.array(EMB.template) + [50, 60])
    x = emb_input_tensor(crop)
    assert crop.shape == (112, 112, 3) and x.shape == (1, 112, 112, 3) and x.min() >= -1 and x.max() <= 1


def _synth_outputs(stride=8, anchor=2 * (10 * 80 + 20) + 1, score=0.9):
    outs = {}
    for s in DET.strides:
        n = (640 // s) ** 2 * 2
        outs[f"score_{s}"] = np.zeros(n, np.float32)
        outs[f"bbox_{s}"] = np.zeros((n, 4), np.float32)
        outs[f"kps_{s}"] = np.zeros((n, 10), np.float32)
    outs[f"score_{stride}"][anchor] = score
    outs[f"bbox_{stride}"][anchor] = [2, 3, 4, 5]          # in stride units
    outs[f"kps_{stride}"][anchor] = [1, 1, -1, 1, 0, 0, 1, -1, -1, -1]
    return outs


def test_decode_scrfd_geometry():
    d = decode_scrfd(_synth_outputs(), conf_thr=0.5)
    assert d.shape == (1, 15)
    cx, cy = 20 * 8, 10 * 8
    assert np.allclose(d[0, :5], [cx - 16, cy - 24, cx + 32, cy + 40, 0.9])
    assert np.allclose(d[0, 5:7], [cx + 8, cy + 8])


def test_decode_applies_sigmoid_to_logits_and_threshold():
    assert len(decode_scrfd(_synth_outputs(score=3.0), 0.5)) == 1  # sigmoid(3)=0.95
    assert len(decode_scrfd(_synth_outputs(score=0.3), 0.5)) == 0


def test_nms_and_to_original():
    a = np.array([[0, 0, 100, 100, .9] + [0] * 10, [5, 5, 105, 105, .8] + [0] * 10, [300, 300, 400, 400, .7] + [0] * 10], np.float32)
    k = nms(a, 0.45)
    assert len(k) == 2 and np.isclose(k[0, 4], .9) and np.isclose(k[1, 4], .7)
    o = to_original(k, 0.5, 1280, 960)
    assert np.allclose(o[0, :4], [0, 0, 200, 200])


def test_preprocess_folder_and_load_outputs(tmp_path):
    imgs = tmp_path / "imgs"; imgs.mkdir()
    cv2.imwrite(str(imgs / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    meta = preprocess_folder(imgs, tmp_path / "det_inputs")
    assert meta["items"][0]["raw"] == "a.raw" and np.isclose(meta["items"][0]["scale"], 3.2)
    assert (tmp_path / "det_inputs" / "a.raw").stat().st_size == 640 * 640 * 3 * 4
    r = tmp_path / "Result_0"; r.mkdir()
    for i in range(9):
        np.zeros(4, np.float32).tofile(r / f"out{i}.raw")
    assert set(load_det_outputs(r)) == set(DET.output_names)


def test_emb_tensor_nchw_for_lvface():
    from frtool.config import LVFACE
    crop = np.random.randint(0, 255, (112, 112, 3), np.uint8)
    x = emb_input_tensor(crop, LVFACE.layout)
    assert x.shape == (1, 3, 112, 112) and x.flags["C_CONTIGUOUS"]
    assert np.allclose(x[0].transpose(1, 2, 0), emb_input_tensor(crop)[0])
