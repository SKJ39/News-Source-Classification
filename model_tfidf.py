"""CIS 5190 Project B: fixed-dimensional hashed word+character TF-IDF classifier.
Only model weights are loaded from the submitted PyTorch state_dict.
"""
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack

WORD_DIM = 2**16
CHAR_DIM = 2**17
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 6)

class NewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = HashingVectorizer(n_features=WORD_DIM, alternate_sign=False,
            ngram_range=WORD_NGRAM_RANGE, token_pattern=r'(?u)\b\w+\b', norm='l2')
        self.char = HashingVectorizer(n_features=CHAR_DIM, alternate_sign=False,
            analyzer='char', ngram_range=CHAR_NGRAM_RANGE, norm='l2')
        self.linear = nn.Linear(WORD_DIM + CHAR_DIM, 2)

    def _transform(self, batch):
        texts = [str(x) for x in batch]
        if not texts:
            return None
        return hstack([self.word.transform(texts), self.char.transform(texts)], format='csr')

    @torch.no_grad()
    def forward(self, batch):
        features = self._transform(batch)
        if features is None:
            return torch.empty((0, 2), dtype=self.linear.weight.dtype)

        # Sparse multiplication avoids creating huge dense feature tensors.
        scores = features @ self.linear.weight.detach().cpu().numpy().T
        scores = np.asarray(scores, dtype=np.float32) + self.linear.bias.detach().cpu().numpy()
        return torch.from_numpy(scores)

    @torch.no_grad()
    def predict(self, batch):
        return self.forward(batch).argmax(dim=1).cpu().numpy().astype(int).tolist()

def get_model():
    return NewsClassifier()
