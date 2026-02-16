from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from .hybrid import HybridTokenizer
from .config import TokenizerConfig, DictConfig


class TokenOut(BaseModel):
    surface: str
    pos: str
    feature: str
    start: int
    end: int
    total_cost: int
    source: str


class TokenizeRequest(BaseModel):
    text: str
    neural_ckpt_dir: Optional[str] = None
    enable_neural_fallback: bool = True


def create_app() -> FastAPI:
    app = FastAPI(title="Japanese Tokenizer", version="1.0.0")

    @app.get("/")
    def root() -> dict:
        return {
            "name": "Hybrid Japanese Tokenizer",
            "docs": "/docs",
            "health": "/health",
            "tokenize": "/tokenize",
        }


    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    # TODO does ts work in gui?
    @app.post("/tokenize", response_model=list[TokenOut])
    def tokenize(req: TokenizeRequest) -> List[TokenOut]:
        cfg = TokenizerConfig(enable_neural_fallback=req.enable_neural_fallback)
        tk = HybridTokenizer(
            dict_cfg=DictConfig(),
            cfg=cfg,
            neural_ckpt_dir=Path(req.neural_ckpt_dir) if req.neural_ckpt_dir else None,
        )
        toks = tk.tokenize(req.text)
        return [TokenOut(**t.__dict__) for t in toks]

    return app
