"""CIS 5190 Project B: ensemble of a TF-IDF linear model, a text CNN and an n-gram bag MLP.

Everything lives in one nn.Module so the submission is a single model.py plus a
single model.pt state_dict. Inputs are the headline slugs produced by
preprocess.py (e.g. "/some-headline-words").

Tokens are mapped to embedding rows with zlib.crc32 (deterministic across
processes, unlike Python's hash()), so no vocabulary file is needed.
"""
import re
import zlib
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn

# TF-IDF linear part (identical features to model_tfidf.py).
WORD_DIM = 2**16
CHAR_DIM = 2**17

# Neural parts.
WORD_BUCKETS = 2**16
NGRAM_BUCKETS = 2**17
CHAR_VOCAB = 257
MAX_WORDS = 40
MAX_CHARS = 200
CNN_WORD_DIM = 128
CNN_CHAR_DIM = 48
CNN_FILTERS = 128
CNN_WORD_KERNELS = (1, 2, 3)
CNN_CHAR_KERNELS = (3, 4, 5)
BAG_DIM = 64
BAG_HIDDEN = 64


def tokens(text: str) -> List[str]:
    return re.findall(r"[^\W_]+", str(text).casefold())


def _bucket(s: str, n: int) -> int:
    """Map a string to 1..n-1 (0 is padding)."""
    return zlib.crc32(s.encode("utf-8")) % (n - 1) + 1


def encode(text: str) -> Tuple[List[int], List[int], List[int]]:
    """Return (word ids, char ids, n-gram ids) for one input string."""
    words = tokens(text)[:MAX_WORDS]
    wids = [_bucket("w:" + w, WORD_BUCKETS) for w in words]
    joined = " ".join(words)[:MAX_CHARS]
    cids = [min(ord(ch), CHAR_VOCAB - 2) + 1 for ch in joined]
    grams = ["u:" + w for w in words]
    grams += ["b:" + a + " " + b for a, b in zip(words, words[1:])]
    padded = "<" + joined + ">"
    for n in (3, 4, 5):
        grams += ["c:" + padded[i:i + n] for i in range(len(padded) - n + 1)]
    nids = [_bucket(g, NGRAM_BUCKETS) for g in grams] or [0]
    return wids, cids, nids


def collate(encoded: Sequence[Tuple[List[int], List[int], List[int]]]):
    """Pad a list of encode() outputs into tensors."""
    b = len(encoded)
    wl = max(max((len(e[0]) for e in encoded), default=0), max(CNN_WORD_KERNELS))
    cl = max(max((len(e[1]) for e in encoded), default=0), max(CNN_CHAR_KERNELS))
    W = torch.zeros((b, wl), dtype=torch.long)
    C = torch.zeros((b, cl), dtype=torch.long)
    wlen = torch.zeros(b, dtype=torch.long)
    clen = torch.zeros(b, dtype=torch.long)
    flat, offsets = [], []
    for i, (wids, cids, nids) in enumerate(encoded):
        if wids:
            W[i, :len(wids)] = torch.tensor(wids)
        if cids:
            C[i, :len(cids)] = torch.tensor(cids)
        wlen[i], clen[i] = len(wids), len(cids)
        offsets.append(len(flat))
        flat.extend(nids)
    return W, wlen, C, clen, torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class WindowConv(nn.Module):
    """1D convolution written as unfold + matmul (identical maths to nn.Conv1d, much
    faster on CPU), followed by ReLU and max-pooling over the valid positions."""

    def __init__(self, in_dim: int, out_dim: int, k: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k * in_dim, out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape                                  # x is (B, L, D), L >= k
        win = x.unfold(1, self.k, 1).transpose(2, 3).reshape(b, length - self.k + 1, self.k * d)
        h = F.relu(self.proj(win))                              # (B, L-k+1, F)
        valid = (lengths - self.k + 1).clamp(min=1)
        mask = torch.arange(h.size(1)).unsqueeze(0) < valid.unsqueeze(1)
        return h.masked_fill(~mask.unsqueeze(2), float("-inf")).max(dim=1).values


class TextCNN(nn.Module):
    """Word-level and character-level 1D convolutions with max-pooling."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.wemb = nn.Embedding(WORD_BUCKETS, CNN_WORD_DIM, padding_idx=0, sparse=True)
        self.cemb = nn.Embedding(CHAR_VOCAB, CNN_CHAR_DIM, padding_idx=0)
        self.wconv = nn.ModuleList([WindowConv(CNN_WORD_DIM, CNN_FILTERS, k) for k in CNN_WORD_KERNELS])
        self.cconv = nn.ModuleList([WindowConv(CNN_CHAR_DIM, CNN_FILTERS, k) for k in CNN_CHAR_KERNELS])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(CNN_FILTERS * (len(CNN_WORD_KERNELS) + len(CNN_CHAR_KERNELS)), 2)

    def forward(self, W, wlen, C, clen, *_):
        we = self.drop(self.wemb(W))
        ce = self.drop(self.cemb(C))
        feats = [conv(we, wlen) for conv in self.wconv] + [conv(ce, clen) for conv in self.cconv]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class BagMLP(nn.Module):
    """fastText-style mean of hashed word/bigram/char n-gram embeddings, then an MLP."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.EmbeddingBag(NGRAM_BUCKETS, BAG_DIM, mode="mean", padding_idx=0, sparse=True)
        self.hidden = nn.Linear(BAG_DIM, BAG_HIDDEN)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(BAG_HIDDEN, 2)

    def forward(self, W, wlen, C, clen, flat, offsets):
        h = self.emb(flat, offsets)
        return self.out(self.drop(F.relu(self.hidden(self.drop(h)))))


class NewsClassifier(nn.Module):
    """Probability-averaging ensemble of TF-IDF logistic regression, TextCNN and BagMLP."""

    def __init__(self):
        super().__init__()
        self.word = HashingVectorizer(n_features=WORD_DIM, alternate_sign=False,
                                      ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", norm="l2")
        self.char = HashingVectorizer(n_features=CHAR_DIM, alternate_sign=False,
                                      analyzer="char", ngram_range=(2, 6), norm="l2")
        self.linear = nn.Linear(WORD_DIM + CHAR_DIM, 2)
        self.cnn = TextCNN()
        self.bag = BagMLP()
        # Mixing weights for [linear, cnn, bag]; set by ensemble.py.
        self.register_buffer("mix", torch.tensor([1.0, 0.0, 0.0]))

    def _linear_logits(self, texts: List[str]) -> torch.Tensor:
        X = hstack([self.word.transform(texts), self.char.transform(texts)], format="csr")
        scores = X @ self.linear.weight.detach().cpu().numpy().T
        return torch.from_numpy(np.asarray(scores, dtype=np.float32)) + self.linear.bias.detach().float()

    @torch.no_grad()
    def forward(self, batch):
        texts = [str(x) for x in batch]
        if not texts:
            return torch.empty((0, 2))
        mix = self.mix.float()
        probs = mix[0] * F.softmax(self._linear_logits(texts), dim=1)
        if mix[1] > 0 or mix[2] > 0:
            inputs = collate([encode(t) for t in texts])
            if mix[1] > 0:
                probs = probs + mix[1] * F.softmax(self.cnn(*inputs).float(), dim=1)
            if mix[2] > 0:
                probs = probs + mix[2] * F.softmax(self.bag(*inputs).float(), dim=1)
        return torch.log(probs.clamp_min(1e-9))

    @torch.no_grad()
    def predict(self, batch):
        return self.forward(batch).argmax(dim=1).cpu().numpy().astype(int).tolist()


def get_model():
    return NewsClassifier()
