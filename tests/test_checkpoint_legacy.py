import json
import pytest

from jp_tokenizer.neural.io import NeuralCheckpoint
from jp_tokenizer.neural.model import BoundaryLSTM


def test_checkpoint_legacy_infer(tmp_path):
    vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2}
    model = BoundaryLSTM(vocab_size=len(vocab), emb_dim=12, hidden_size=20, dropout=0.3)
    ckpt = NeuralCheckpoint(tmp_path)

    ckpt.save(model, vocab)
    vocab_path = tmp_path / "vocab.json"
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    data.pop("meta", None)
    vocab_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.warns(UserWarning):
        loaded_model, loaded_vocab, meta = ckpt.load()

    assert loaded_vocab == vocab
    assert loaded_model.emb.embedding_dim == 12
    assert loaded_model.lstm.hidden_size == 20
    assert meta["emb_dim"] == 12
    assert meta["hidden_size"] == 20
