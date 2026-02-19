import torch
import pytest

from jp_tokenizer.config import DictConfig, TokenizerConfig
from jp_tokenizer.hybrid import HybridTokenizer


class DummyModel:
    def predict_logprobs(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Prefer S everywhere to yield single-char tokens.
        T = x.shape[1]
        logp = torch.full((T, 4), -2.0)
        logp[:, 3] = -0.1
        return logp.unsqueeze(0)


def test_neural_cost_segment_cumulative():
    tk = HybridTokenizer(dict_cfg=DictConfig(auto_download=False), cfg=TokenizerConfig(force_neural=True))
    tk._neural = DummyModel()
    tk._vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2}
    tk._vocab_meta = {"unk_id": 1}
    tk._dict_ready = False

    toks = tk._neural_segment("aa", 0, 2)
    assert [t.surface for t in toks] == ["a", "a"]
    assert toks[0].total_cost == pytest.approx(0.1)
    assert toks[1].total_cost == pytest.approx(0.2)
