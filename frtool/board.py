"""SSH/SFTP session to the board; runs qnn-net-run in /tmp on the board only."""
from __future__ import annotations

import posixpath
import stat
import time
from pathlib import Path

import paramiko

from .config import BOARD, BoardConfig


class Board:
    def __init__(self, cfg: BoardConfig = BOARD, timeout: float = 15.0):
        self.cfg = cfg
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(cfg.host, port=cfg.port, username=cfg.user, password=cfg.password,
                         timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
        self.sftp = self.ssh.open_sftp()

    def close(self):
        try:
            self.sftp.close()
        finally:
            self.ssh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def sh(self, cmd: str, timeout: float = 600.0, check: bool = True) -> tuple[int, str, str]:
        _, out, err = self.ssh.exec_command(cmd, timeout=timeout)
        o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
        rc = out.channel.recv_exit_status()
        if check and rc != 0:
            raise RuntimeError(f"board command failed rc={rc}: {cmd}\n--- stdout ---\n{o}\n--- stderr ---\n{e}")
        return rc, o, e

    # ---------- files ----------
    def put_dir(self, local: Path, remote: str, pattern: str = "*.raw") -> list[str]:
        self.sh(f"mkdir -p {remote}")
        names = []
        for f in sorted(Path(local).glob(pattern)):
            self.sftp.put(str(f), posixpath.join(remote, f.name))
            names.append(f.name)
        return names

    def get_tree(self, remote: str, local: Path) -> None:
        local = Path(local); local.mkdir(parents=True, exist_ok=True)
        for entry in self.sftp.listdir_attr(remote):
            r = posixpath.join(remote, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                self.get_tree(r, local / entry.filename)
            else:
                self.sftp.get(r, str(local / entry.filename))

    def write_text(self, remote: str, text: str) -> None:
        with self.sftp.open(remote, "w") as f:
            f.write(text)

    # ---------- inference ----------
    def run_model(self, model_so: str, input_name: str, raw_names: list[str],
                  remote_dir: str, log_level: str = "error") -> str:
        """Runs qnn-net-run on the DSP over raw_names (already uploaded to remote_dir/inputs).
        model_so: file name in cfg.model_dir, or an absolute path on the board.
        Returns the remote output dir (contains Result_0..Result_{n-1})."""
        cfg = self.cfg
        model_path = model_so if model_so.startswith("/") else posixpath.join(cfg.model_dir, model_so)
        in_dir = posixpath.join(remote_dir, "inputs")
        out_dir = posixpath.join(remote_dir, "outputs")
        list_path = posixpath.join(remote_dir, "input_list.txt")
        self.write_text(list_path, "".join(f"{input_name}:={posixpath.join(in_dir, n)}\n" for n in raw_names))
        self.sh(f"rm -rf {out_dir}; mkdir -p {out_dir}")
        cmd = (f"{cfg.env_prefix} cd {remote_dir} && {cfg.net_run}"
               f" --model {model_path} --backend {cfg.backend_path}"
               f" --input_list {list_path} --output_dir {out_dir} --log_level {log_level}"
               f" > {remote_dir}/net_run.log 2>&1")
        t0 = time.time()
        rc, _, _ = self.sh(cmd, check=False)
        dt = time.time() - t0
        _, log, _ = self.sh(f"tail -n 40 {remote_dir}/net_run.log", check=False)
        if rc != 0:
            raise RuntimeError(f"qnn-net-run failed (rc={rc}) for {model_so}:\n{log}")
        print(f"[board] {posixpath.basename(model_so)}: {len(raw_names)} inputs in {dt:.1f}s")
        return out_dir

    def cleanup(self, remote_dir: str) -> None:
        if remote_dir.startswith(self.cfg.work_root):
            self.sh(f"rm -rf {remote_dir}", check=False)
