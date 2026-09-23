# fr_tool — Face detection + Face embedding trên board QRB5165

Tool chạy **ở máy host** (Linux/WSL). Nó tiền xử lý ảnh, đẩy lên board, gọi `qnn-net-run`
trên cDSP với 2 model có sẵn, kéo kết quả về và hậu xử lý.

- **Face detection**: `libscrfd_25g_kps_w8a8.so` → bbox, 5 landmark (kps), confidence
- **Face embedding**: `libarcface_mnet_w8a8.so` → vector 512 chiều đã L2-normalize

**Không cần cài hay copy gì lên board.** Tool chỉ dùng những thứ đã có sẵn trên board và
mọi file tạm đều nằm trong `/tmp/fr_tool/<session>/`, tự xoá sau khi chạy xong.

---

## 1. Yêu cầu

### Board (đã có sẵn, không cần làm gì)

| Thứ | Vị trí trên board |
|---|---|
| IP / tài khoản | `192.168.138.142`, `root` / `oelinux123` |
| Model | `/VQEC_V2/models/face-recognition/qrb5165/libscrfd_25g_kps_w8a8.so` và `libarcface_mnet_w8a8.so` |
| Runtime | `/VQEC_V2/runtime/qairt/2.28.2.241116/` (`qnn-net-run`, `libQnnDsp.so`, skel Hexagon v66) |
| File tạm | `/tmp/fr_tool/` |

Tất cả được hardcode trong `frtool/config.py`. Đổi IP / mật khẩu / đường dẫn ở đó nếu cần.

### Host

- Python 3.9+
- Máy host ping được board (`ping 192.168.138.142`)
- Thư viện Python: `numpy`, `opencv-python`, `paramiko` (ssh/scp bằng Python, nên **không cần `sshpass`**)

---

## 2. Cài đặt (làm 1 lần)

```bash
cd embedding_call
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu đã có venv QAIRT (`~/qairt/venv-2.43`) thì dùng luôn, không cần cài gì:

```bash
~/qairt/venv-2.43/bin/python fr_tool.py --help
```

Kiểm tra nhanh (chỉ test logic trên host, không cần board):

```bash
python -m pytest tests
```

---

## 3. Chạy nhanh (1 lệnh)

```bash
python fr_tool.py all <thư_mục_ảnh> --work work --draw
```

Ví dụ với 3 ảnh mẫu đi kèm:

```bash
python fr_tool.py all sample_images --work work --draw
```

Log mong đợi:

```
[preprocess] 3 images -> work/det_inputs
[board] libscrfd_25g_kps_w8a8.so: 3 inputs in 0.7s
[board] libarcface_mnet_w8a8.so: 52 inputs in 1.6s
[run] remote session /tmp/fr_tool/20260911_092649 cleaned; outputs in work
[postprocess] 3 images, 52 faces -> work/results.json
[postprocess] visualizations -> work/vis
```

Ảnh hỗ trợ: `.jpg .jpeg .png .bmp .webp` (không đệ quy vào thư mục con).

---

## 4. Chạy từng bước

Khi muốn kiểm tra file `.raw`, chạy lại board nhiều lần, hay đổi ngưỡng mà không chạy lại board.

```bash
# Bước 1: ảnh -> input .raw cho detector (chỉ ở host)
python fr_tool.py preprocess <thư_mục_ảnh> --work work
#   -> work/det_inputs/<tên_ảnh>.raw  (float32, NHWC 1x640x640x3)
#   -> work/det_inputs/meta.json      (kích thước gốc, tỉ lệ scale)

# Bước 2: đẩy lên board, chạy 2 model, kéo output về (cần board)
python fr_tool.py run --work work
#   -> work/det_outputs/Result_<i>/out0..out8.raw   (detector, float32)
#   -> work/emb_inputs/*.raw, work/crops/*.png       (mặt đã căn chỉnh 112x112)
#   -> work/emb_outputs/Result_<k>/embedding.raw     (512 float32)
#   -> work/detections.json

# Bước 3: hậu xử lý -> results.json (+ ảnh vẽ bbox/kps) (chỉ ở host)
python fr_tool.py postprocess --work work --draw
#   -> work/results.json, work/vis/*.jpg
```

Lưu ý: bước `run` bắt buộc phải decode detection ngay trên host để biết mặt ở đâu
rồi mới tạo input cho model embedding, nên bên trong nó là:
detect → decode → align/crop → embed.

---

## 5. Tuỳ chọn

| Cờ | Lệnh | Mặc định | Ý nghĩa |
|---|---|---|---|
| `--work DIR` | tất cả | `./work` | Thư mục làm việc trên host |
| `--conf X` | run, postprocess, all | `0.5` | Ngưỡng confidence của detector |
| `--iou X` | run, postprocess, all | `0.45` | Ngưỡng IoU cho NMS |
| `--keep-remote` | run, all | tắt | Giữ lại `/tmp/fr_tool/<session>` trên board để debug |
| `--emb-mode M` | run, all | `arcface` | Kiểu I/O của model embedding: `arcface` hoặc `lvface` |
| `--emb-model F` | run, all | `libarcface_mnet_w8a8.so` | File `.so` embedding. Nếu là file có trên host → tự upload lên board. Nếu không → tên file trong thư mục model của board, hoặc đường dẫn tuyệt đối trên board. Bắt buộc khi `--emb-mode lvface` |
| `--output FILE` | postprocess, all | `<work>/results.json` | Đường dẫn file kết quả |
| `--draw` | postprocess, all | tắt | Ghi ảnh vẽ bbox/kps vào `<work>/vis/` |

Ví dụ hạ ngưỡng để bắt mặt nhỏ/mờ hơn, và giữ file trên board:

```bash
python fr_tool.py all my_photos --work out_photos --conf 0.3 --keep-remote --draw
```

Chạy embedding bằng LVFace thay cho ArcFace (file `.so` nằm trên host, tool tự upload lên
`/tmp/fr_tool/<session>` và xoá sau khi chạy):

```bash
python fr_tool.py all my_photos --work out_lv --draw \
    --emb-mode lvface --emb-model /mnt/d/Models/LVFace/2.28.2.241116/libLVFace_dlc_dynamic.so
```

Mode `lvface`: input `data` NCHW 1x3x112x112 (cùng crop căn chỉnh 112x112, cùng chuẩn hoá `x/127.5-1`),
output `_964` 512 chiều. Ngưỡng cosine `0.41` là của ArcFace, LVFace cần tự chỉnh lại.
Bước `postprocess` tự biết mode từ `work/emb_inputs/meta.json`, không cần truyền lại cờ.

---

## 6. Định dạng kết quả `results.json`

Một phần tử cho mỗi ảnh; toạ độ theo **pixel ảnh gốc**.

```json
[
  {
    "image": "/abs/path/photo.jpg",
    "width": 1024,
    "height": 768,
    "faces": [
      {
        "bbox": [584.24, 229.56, 612.51, 263.78],
        "confidence": 0.8876,
        "kps": [[591.79, 240.16], [603.1, 239.9], [597.6, 247.3], [592.4, 255.1], [602.2, 254.9]],
        "embedding": [0.0123, -0.0456, "... 512 số, đã L2-normalize ..."]
      }
    ]
  }
]
```

- `bbox`: `[x1, y1, x2, y2]`
- `kps`: 5 điểm theo thứ tự mắt trái, mắt phải, mũi, khoé miệng trái, khoé miệng phải
- `embedding`: so sánh 2 mặt bằng cosine = tích vô hướng (đã chuẩn hoá). Hệ thống VQEC dùng ngưỡng `0.41`.

Đọc bằng Python:

```python
import json, numpy as np
r = json.load(open("work/results.json"))
e1 = np.array(r[0]["faces"][0]["embedding"])
e2 = np.array(r[1]["faces"][0]["embedding"])
print("cosine:", float(e1 @ e2))
```

---

## 7. Pipeline bên trong

| Bước | Chạy ở | Chi tiết |
|---|---|---|
| Tiền xử lý detector | host | Letterbox góc trên-trái về 640x640, RGB, `(x-127.5)/128`, float32 NHWC |
| Detect | board (cDSP) | `qnn-net-run --model libscrfd_25g_kps_w8a8.so --backend libQnnDsp.so` |
| Decode | host | `out0..out8` = score / bbox / kps ở stride 8/16/32, 2 anchor mỗi ô, sigmoid nếu là logit, NMS |
| Căn chỉnh mặt | host | Similarity transform 5 landmark → template ArcFace 112x112, `x/127.5-1` |
| Embed | board (cDSP) | `qnn-net-run --model libarcface_mnet_w8a8.so` → `embedding.raw` |
| Hậu xử lý | host | L2-normalize embedding, ghi `results.json`, vẽ `vis/` |

`qnn-net-run` nhận input float32 và tự lượng tử hoá theo encoding trong `.so`; output cũng
được trả về dạng float32 đã de-quantize, nên host không cần biết scale/offset.
Cách tiền xử lý và decode lấy theo đúng mã `ai_worker` của VQEC trên board
(`/VQEC_V2/apps/edge-camera/backend/bootstrap/ai_worker/src`).

---

## 8. Cấu trúc thư mục

```
embedding_call/
├── fr_tool.py            # entrypoint CLI
├── frtool/
│   ├── config.py         # IP board, tài khoản, đường dẫn model/runtime, ngưỡng
│   ├── preprocess.py     # ảnh -> .raw; căn chỉnh mặt
│   ├── board.py          # ssh/sftp, gọi qnn-net-run, dọn /tmp trên board
│   ├── postprocess.py    # decode SCRFD, NMS, chuẩn hoá embedding
│   └── cli.py            # các lệnh preprocess / run / postprocess / all
├── tests/                # unit test (pytest, không cần board)
├── sample_images/        # 3 ảnh mẫu để chạy thử
├── requirements.txt
└── work/                 # sinh ra khi chạy (có thể xoá)
```

---

## 9. Lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `TimeoutError: timed out` khi connect | Board chưa lên mạng hoặc rớt mạng tạm thời. `ping 192.168.138.142` rồi chạy lại. Board thỉnh thoảng mất kết nối khoảng 1 phút. |
| `qnn-net-run failed (rc=...)` | Xem log in ra ngay sau đó. Chạy lại với `--keep-remote` rồi xem `/tmp/fr_tool/<session>/det/net_run.log` trên board. |
| `ModuleNotFoundError: cv2 / paramiko` | Chưa `pip install -r requirements.txt` hoặc đang dùng sai Python. |
| Ảnh không thấy mặt nào | Hạ `--conf` (ví dụ `0.3`). Mặt quá nhỏ (dưới ~10 px sau khi resize về 640) sẽ không bắt được. |
| Chạy chậm với nhiều ảnh | Mỗi ảnh gửi lên là 4.9 MB float32. Tách folder nhỏ hơn, hoặc cân nhắc chuyển sang input uint8 native (chưa hỗ trợ). |
| Trên board còn rác `/tmp/fr_tool/...` | Do dùng `--keep-remote` hoặc bị ngắt giữa chừng. Xoá bằng `ssh root@192.168.138.142 rm -rf /tmp/fr_tool/*`. |
