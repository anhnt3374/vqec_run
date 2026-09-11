"""Hard-coded board / model configuration."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoardConfig:
    host: str = "192.168.138.142"
    user: str = "root"
    password: str = "oelinux123"
    port: int = 22
    model_dir: str = "/VQEC_V2/models/face-recognition/qrb5165"
    det_model: str = "libscrfd_25g_kps_w8a8.so"
    emb_model: str = "libarcface_mnet_w8a8.so"
    qairt_root: str = "/VQEC_V2/runtime/qairt/2.28.2.241116"
    backend: str = "libQnnDsp.so"          # cDSP (Hexagon v66) on QRB5165
    work_root: str = "/tmp/fr_tool"        # everything on the board lives here

    @property
    def net_run(self) -> str:
        return f"{self.qairt_root}/bin/qnn-net-run"

    @property
    def backend_path(self) -> str:
        return f"{self.qairt_root}/lib/aarch64-ubuntu-gcc9.4/{self.backend}"

    @property
    def env_prefix(self) -> str:
        ld = f"{self.qairt_root}/lib/aarch64-ubuntu-gcc9.4:/usr/lib:/usr/lib/aarch64-linux-gnu"
        # FastRPC wants ';' separators here (same convention as snpe-net-run).
        adsp = f"{self.qairt_root}/lib/hexagon-v66/unsigned;/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp;/usr/lib/dsp"
        return f"export LD_LIBRARY_PATH={ld}; export ADSP_LIBRARY_PATH='{adsp}';"


@dataclass(frozen=True)
class DetConfig:
    input_name: str = "images"
    size: int = 640                         # NHWC 1x640x640x3, RGB, (x-127.5)/128
    strides: tuple = (8, 16, 32)
    anchors_per_cell: int = 2
    conf_threshold: float = 0.5
    nms_iou: float = 0.45
    max_faces: int = 100
    # qnn-net-run writes out0..out8; positional mapping from the board's QnnOutputOrder.cpp
    output_names: tuple = ("score_8", "score_16", "score_32",
                           "bbox_8", "bbox_16", "bbox_32",
                           "kps_8", "kps_16", "kps_32")


@dataclass(frozen=True)
class EmbConfig:
    input_name: str = "input_1"
    output_name: str = "embedding"
    size: int = 112                         # NHWC 1x112x112x3, RGB, x/127.5-1
    dim: int = 512
    # ArcFace 112x112 5-point template (same values as FaceAlignment.cpp on the board)
    template: tuple = field(default_factory=lambda: (
        (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366),
        (41.5493, 92.3655), (70.7299, 92.2041)))


BOARD = BoardConfig()
DET = DetConfig()
EMB = EmbConfig()
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
