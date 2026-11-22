"""Linear SVM classifier with fixed-dimensional word and character features."""

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn

WORD_DIM = 2**16
CHAR_DIM = 2**17
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 6)


class NewsClassifier(nn.Module):
    """Expose the fitted linear SVM as a PyTorch-compatible classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.word = HashingVectorizer(n_features=WORD_DIM, alternate_sign=False, ngram_range=WORD_NGRAM_RANGE, 
                                      token_pattern=r"(?u)\b\w+\b", norm="l2")
        self.char = HashingVectorizer(n_features=CHAR_DIM, alternate_sign=False, analyzer="char",
                                      ngram_range=CHAR_NGRAM_RANGE, norm="l2")
        self.linear = nn.Linear(WORD_DIM + CHAR_DIM, 2)

    def _transform(self, batch: list[object]):
        texts = [str(text) for text in batch]
        if not texts:
            return None
        word_features = self.word.transform(texts)
        char_features = self.char.transform(texts)
        return hstack([word_features, char_features], format="csr")

    @torch.no_grad()
    def forward(self, batch: list[object]) -> torch.Tensor:
        features = self._transform(batch)
        if features is None:
            return torch.empty((0, 2), dtype=self.linear.weight.dtype)

        scores = features @ self.linear.weight.detach().cpu().numpy().T
        scores = np.asarray(scores, dtype=np.float32)
        scores += self.linear.bias.detach().cpu().numpy()
        return torch.from_numpy(scores)

    @torch.no_grad()
    def predict(self, batch: list[object]) -> list[int]:
        return self.forward(batch).argmax(dim=1).cpu().tolist()


def get_model() -> NewsClassifier:
    return NewsClassifier()