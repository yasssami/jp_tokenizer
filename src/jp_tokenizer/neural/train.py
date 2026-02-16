from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .model import BoundaryBatch, BoundaryLSTM
from .io import NeuralCheckpoint


PAD_ID = 0
UNK_ID = 1


class CharDataset(Dataset):
    def __init__(self, examples: List[Tuple[List[int], List[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.examples[idx]


def build_vocab(examples_chars: Iterable[List[str]], max_size: int = 40000) -> Dict[str, int]:
    from collections import Counter
    c = Counter()
    for chars in examples_chars:
        c.update(chars)
    vocab: Dict[str, int] = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    for ch, _ in c.most_common(max_size - len(vocab)):
        if ch in vocab:
            continue
        vocab[ch] = len(vocab)
    return vocab


def encode_examples(examples: Iterable, vocab: Dict[str, int]) -> List[Tuple[List[int], List[int]]]:
    out: List[Tuple[List[int], List[int]]] = []
    for ex in examples:
        chars = ex.chars
        labels = ex.labels
        x = [vocab.get(ch, UNK_ID) for ch in chars]
        y = list(labels)
        if len(x) != len(y) or len(x) == 0:
            continue
        out.append((x, y))
    return out


def collate(batch: List[Tuple[List[int], List[int]]]) -> BoundaryBatch:
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    T = int(lengths.max().item())
    B = len(batch)
    x = torch.full((B, T), PAD_ID, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    for i, (xi, yi) in enumerate(batch):
        x[i, : len(xi)] = torch.tensor(xi, dtype=torch.long)
        y[i, : len(yi)] = torch.tensor(yi, dtype=torch.long)
    return BoundaryBatch(x=x, y=y, lengths=lengths)


def label_smoothed_ce(logits: torch.Tensor, target: torch.Tensor, eps: float = 0.05) -> torch.Tensor:
    """
    logits: (B,T,C), target: (B,T) with -100 ignored
    """
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    target = target.view(B * T)
    mask = target != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    logits_m = logits[mask]
    target_m = target[mask]
    logp = F.log_softmax(logits_m, dim=-1)
    nll = F.nll_loss(logp, target_m, reduction="mean")
    smooth = -logp.mean(dim=-1).mean()
    return (1 - eps) * nll + eps * smooth


@dataclass(frozen=True)
class TrainConfig:
    emb_dim: int = 128
    hidden: int = 256
    dropout: float = 0.1
    epochs: int = 3
    batch_size: int = 64
    lr: float = 2e-3
    grad_clip: float = 1.0
    vocab_max: int = 40000
    device: str = "cpu"
    label_smooth: float = 0.05


def train_boundary_model(
    examples_train: Iterable,
    examples_dev: Iterable,
    out_dir: Path,
    cfg: TrainConfig = TrainConfig(),
) -> None:
    # build vocab from train
    train_chars = (ex.chars for ex in examples_train)
    # materialize train for reuse
    examples_train = list(examples_train) if not isinstance(examples_train, list) else examples_train
    examples_dev = list(examples_dev) if not isinstance(examples_dev, list) else examples_dev

    vocab = build_vocab((ex.chars for ex in examples_train), max_size=cfg.vocab_max)
    train_encoded = encode_examples(examples_train, vocab)
    dev_encoded = encode_examples(examples_dev, vocab)

    ds_train = CharDataset(train_encoded)
    ds_dev = CharDataset(dev_encoded)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    dl_dev = DataLoader(ds_dev, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = BoundaryLSTM(vocab_size=len(vocab), emb_dim=cfg.emb_dim, hidden_size=cfg.hidden, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_dev = 1e30
    ckpt = NeuralCheckpoint(out_dir)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"train epoch {epoch}", leave=False)
        for batch in pbar:
            batch = BoundaryBatch(
                x=batch.x.to(cfg.device),
                y=batch.y.to(cfg.device),
                lengths=batch.lengths.to(cfg.device),
            )
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = label_smoothed_ce(logits, batch.y, eps=cfg.label_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # dev
        model.eval()
        dev_loss = 0.0
        dev_n = 0
        with torch.no_grad():
            for batch in dl_dev:
                batch = BoundaryBatch(
                    x=batch.x.to(cfg.device),
                    y=batch.y.to(cfg.device),
                    lengths=batch.lengths.to(cfg.device),
                )
                logits = model(batch)
                loss = label_smoothed_ce(logits, batch.y, eps=0.0)
                dev_loss += float(loss.item())
                dev_n += 1
        dev_loss = dev_loss / max(1, dev_n)

        if dev_loss < best_dev:
            best_dev = dev_loss
            ckpt.save(model, vocab)

    # final save (best already saved); ensure dir exists
    if not ckpt.exists():
        ckpt.save(model, vocab)
