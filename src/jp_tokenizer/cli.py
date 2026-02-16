from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from .config import DictConfig, TokenizerConfig
from .dict.downloader import ensure_unidic_mecab
from .hybrid import HybridTokenizer


app = typer.Typer(add_completion=False)


@app.command("download-dict")
def download_dict(
    version: str = typer.Option("2.1.2", help="unidic-mecab tag version"),
) -> None:
    cfg = DictConfig(version=version)
    res = ensure_unidic_mecab(cfg)
    rprint(f"[green]OK[/green] installed dict to: {res.installed_to}")


@app.command("tokenize")
def tokenize(
    text: str,
    neural_ckpt: Optional[Path] = typer.Option(None, help="Path to neural checkpoint dir"),
    no_neural: bool = typer.Option(False, help="toggle neural fallback"),
) -> None:
    cfg = TokenizerConfig(enable_neural_fallback=not no_neural)
    tk = HybridTokenizer(dict_cfg=DictConfig(), cfg=cfg, neural_ckpt_dir=neural_ckpt)
    toks = tk.tokenize(text)
    for t in toks:
        rprint(f"{t.start:>4}-{t.end:<4} {t.source:<6} {t.pos:<18} {t.surface}")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(), host=host, port=port)


@app.command("train-neural")
def train_neural(
    out: Path = typer.Option(Path("model_ckpt"), help="Output dir for checkpoint"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(64),
    device: str = typer.Option("cpu"),
) -> None:
    # train on UD japanese
    from .neural.data_ud import iter_ud_japanese
    from .neural.train import train_boundary_model, TrainConfig
    train_ex = list(iter_ud_japanese("train"))
    try:
        dev_ex = list(iter_ud_japanese("validation"))
        if (len(dev_ex)) == 0:
            raise RuntimeError("empty validation split")
    except Exception:
        dev_ex = train_ex[:2000]
        train_ex = train_ex[2000:]
    cfg = TrainConfig(epochs=epochs, batch_size=batch_size, device=device)
    train_boundary_model(train_ex, dev_ex, out, cfg)
    rprint(f"[green]OK[/green] saved checkpoint to: {out}")


if __name__ == "__main__":
    app()
