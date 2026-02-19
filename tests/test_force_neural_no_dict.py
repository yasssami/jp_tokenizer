from jp_tokenizer.config import DictConfig, TokenizerConfig
from jp_tokenizer.hybrid import HybridTokenizer
from jp_tokenizer.neural.io import NeuralCheckpoint
from jp_tokenizer.neural.model import BoundaryLSTM


def test_force_neural_no_dict(tmp_path):
    vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3}
    model = BoundaryLSTM(vocab_size=len(vocab), emb_dim=8, hidden_size=8, dropout=0.0)
    ckpt_dir = tmp_path / "ckpt"
    NeuralCheckpoint(ckpt_dir).save(model, vocab)

    dict_cfg = DictConfig(root_dir=tmp_path / "dicts", auto_download=False)
    cfg = TokenizerConfig(force_neural=True, enable_neural_fallback=True)
    tk = HybridTokenizer(dict_cfg=dict_cfg, cfg=cfg, neural_ckpt_dir=ckpt_dir)

    toks = tk.tokenize("ab")
    assert toks
    assert all(t.source == "NEURAL" for t in toks)
    assert all(t.pos == "UNK" for t in toks)
