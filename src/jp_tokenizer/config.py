from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from platformdirs import user_data_dir


@dataclass(frozen=True)
class DictConfig:
    name: str = "unidic-mecab"
    version: str = "2.1.2"
    # TODO doublecheck: should be installed under .\jp_tokenizer\dicts\...
    root_dir: Path = Path(user_data_dir("jp_tokenizer", "jpt")) / "dicts"
    auto_download: bool = True

    @property
    def install_dir(self) -> Path:
        return self.root_dir / f"{self.name}-{self.version}"

    @property
    def lex_csv(self) -> Path:
        return self.install_dir / "lex.csv"

    @property
    def matrix_def(self) -> Path:
        return self.install_dir / "matrix.def"

    @property
    def char_def(self) -> Path:
        return self.install_dir / "char.def"

    @property
    def unk_def(self) -> Path:
        return self.install_dir / "unk.def"

    @property
    def left_id_def(self) -> Path:
        return self.install_dir / "left-id.def"

    @property
    def right_id_def(self) -> Path:
        return self.install_dir / "right-id.def"


@dataclass(frozen=True)
class TokenizerConfig:
    # lattice / decoding (check that logic is sound for jp)
    max_word_len: int = 32
    unk_penalty: int = 8000
    # TODO check that fallback is used when necessary
    enable_neural_fallback: bool = True
    force_neural: bool = False
    fallback_cost_per_char_threshold: float = 2500.0
    fallback_expand_chars: int = 2
    fallback_max_span_chars: int = 64
