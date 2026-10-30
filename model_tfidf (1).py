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

class NewsClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = HashingVectorizer(n_features=WORD_DIM, alternate_sign=False,
            ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b', norm='l2')
        self.char = HashingVectorizer(n_features=CHAR_DIM, alternate_sign=False,
            analyzer='char', ngram_range=(3, 5), norm='l2')
        self.linear = nn.Linear(WORD_DIM + CHAR_DIM, 2)

    @torch.no_grad()
    def predict(self, batch):
        texts = [str(x) for x in batch]
        if not texts:
            return []
        features = hstack([self.word.transform(texts), self.char.transform(texts)], format='csr')
        # Sparse multiplication avoids creating huge dense feature tensors.
        scores = features @ self.linear.weight.detach().cpu().numpy().T
        scores = np.asarray(scores) + self.linear.bias.detach().cpu().numpy()
        return scores.argmax(axis=1).astype(int).tolist()

def get_model():
    return NewsClassifier()
