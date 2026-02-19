from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from .hybrid import HybridTokenizer
from .config import TokenizerConfig, DictConfig
from .pos_en import translate_pos, translate_pos_components


class TokenOut(BaseModel):
    surface: str
    pos: str
    pos_en: str
    pos_en_components: List[str]
    feature: str
    start: int
    end: int
    total_cost: float
    source: str


class TokenizeRequest(BaseModel):
    text: str
    neural_ckpt_dir: Optional[str] = None
    enable_neural_fallback: bool = True
    force_neural: bool = False
    auto_download_dict: bool = True

    include_pos_en: bool = True


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
        cfg = TokenizerConfig(enable_neural_fallback=req.enable_neural_fallback, force_neural=req.force_neural)
        tk = HybridTokenizer(
            dict_cfg=DictConfig(auto_download=req.auto_download_dict),
            cfg=cfg,
            neural_ckpt_dir=Path(req.neural_ckpt_dir) if req.neural_ckpt_dir else None,
        )
        toks = tk.tokenize(req.text)
        out: List[TokenOut] = []
        for t in toks:
            if req.include_pos_en:
                pos_en = translate_pos(t.pos)
                pos_en_components = translate_pos_components(t.pos)
            else:
                pos_en = ""
                pos_en_components = []
            out.append(TokenOut(
                surface=t.surface,
                pos=t.pos,
                pos_en=pos_en,
                pos_en_components=pos_en_components,
                feature=t.feature,
                start=t.start,
                end=t.end,
                total_cost=t.total_cost,
                source=t.source,
            ))
        return out

    return app
