from __future__ import annotations
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
import requests
from tqdm import tqdm
from ..config import DictConfig


@dataclass(frozen=True)
class DownloadResult:
    installed_to: Path
    version: str


def _download_bytes(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", "0") or "0")
    buf = io.BytesIO()
    with tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc="ダウンロード中...") as pbar:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue()


def ensure_unidic_mecab(cfg: DictConfig = DictConfig()) -> DownloadResult:
    """
    Downloads and installs lindera/unidic-mecab source files (lex.csv, matrix.def, unk.def, char.def, ...).
    We fetch the GitHub tag tarball for cfg.version and extract its root contents into cfg.install_dir.

    The repository contains exactly the MeCab-style source files we need (lex.csv, matrix.def, left/right-id.def, ...).
    """
    install_dir = cfg.install_dir
    if (install_dir / "lex.csv").exists() and (install_dir / "matrix.def").exists():
        return DownloadResult(installed_to=install_dir, version=cfg.version)

    install_dir.mkdir(parents=True, exist_ok=True)

    # TODO check tarball unpacking accuracy
    url = f"https://github.com/lindera/unidic-mecab/archive/refs/tags/{cfg.version}.tar.gz"
    data = _download_bytes(url)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        root_prefix = None
        for m in tf.getmembers():
            name = m.name
            if name.endswith("/") and root_prefix is None and name.count("/") == 1:
                root_prefix = name
                break
        if root_prefix is None:
            # fallback: infer from first member
            first = tf.getmembers()[0].name
            root_prefix = first.split("/")[0] + "/"

        want = {
            "lex.csv",
            "matrix.def",
            "char.def",
            "unk.def",
            "left-id.def",
            "right-id.def",
            "rewrite.def",
            "dicrc",
        }

        for fname in want:
            member = tf.getmember(root_prefix + fname)
            out_path = install_dir / fname
            f = tf.extractfile(member)
            assert f is not None
            out_path.write_bytes(f.read())

    return DownloadResult(installed_to=install_dir, version=cfg.version)
