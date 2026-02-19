import pytest

from jp_tokenizer.neural.io import NeuralCheckpoint
from jp_tokenizer.neural.model import BoundaryLSTM


def test_checkpoint_meta_roundtrip(tmp_path):
    vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2}
    model = BoundaryLSTM(vocab_size=len(vocab), emb_dim=16, hidden_size=32, dropout=0.2)
    ckpt = NeuralCheckpoint(tmp_path)

    ckpt.save(model, vocab)
    loaded_model, loaded_vocab, meta = ckpt.load()

    assert loaded_vocab == vocab
    assert loaded_model.emb.embedding_dim == 16
    assert loaded_model.lstm.hidden_size == 32
    assert loaded_model.dropout.p == pytest.approx(0.2)
    assert meta["emb_dim"] == 16
    assert meta["hidden_size"] == 32
    assert meta["dropout"] == pytest.approx(0.2)
