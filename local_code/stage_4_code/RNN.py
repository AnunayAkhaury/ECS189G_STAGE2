# RNN.py
"""
RNN.py

Provides data loading, preprocessing, vocabulary building, dataset class, and RNN model for text classification.
"""
import os
import re
import string
from collections import Counter
import json
import torch
from torch.utils.data import Dataset
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────────────────
# 1) Data loading and preprocessing
# ────────────────────────────────────────────────────────────────────────────────
def load_data(dataset_dir, split):
    """
    Reads text files under dataset_dir/split/{pos,neg} and returns lists of texts and labels.
    """
    texts, labels = [], []
    for label in ("pos", "neg"):
        folder = os.path.join(dataset_dir, split, label)
        for fname in os.listdir(folder):
            if fname.endswith('.txt'):
                path = os.path.join(folder, fname)
                with open(path, encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label == 'pos' else 0)
    return texts, labels


def clean_text(text):
    """
    Lowercase, strip HTML tags, remove punctuation.
    """
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# ────────────────────────────────────────────────────────────────────────────────
# 2) Vocabulary building and encoding
# ────────────────────────────────────────────────────────────────────────────────
def build_vocab(texts, max_vocab):
    """
    Build a word2idx mapping of the most common max_vocab-2 words, with 0 for PAD and 1 for UNK.
    """
    counter = Counter()
    for text in texts:
        tokens = clean_text(text).split()
        counter.update(tokens)
    most_common = counter.most_common(max_vocab - 2)
    # Reserve 0 for PAD, 1 for UNK
    word2idx = {w: idx+2 for idx, (w, _) in enumerate(most_common)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    return word2idx


def encode_texts(texts, word2idx, max_len):
    """
    Convert list of raw texts to a tensor of shape (len(texts), max_len) of indices.
    """
    sequences = []
    for text in texts:
        tokens = clean_text(text).split()
        idxs = [word2idx.get(tok, 1) for tok in tokens]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        sequences.append(idxs)
    return torch.tensor(sequences, dtype=torch.long)

# ────────────────────────────────────────────────────────────────────────────────
# 3) PyTorch Dataset
# ────────────────────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.X = encode_texts(texts, word2idx, max_len)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ────────────────────────────────────────────────────────────────────────────────
# 4) RNN Model
# ────────────────────────────────────────────────────────────────────────────────
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_units, num_layers, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(rnn_units * direction_factor, 1)

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        out, _ = self.rnn(emb)   # out: [batch, seq_len, hidden*dir]
        # take last time-step
        last = out[:, -1, :]
        logits = self.fc(last).squeeze(1)
        return logits