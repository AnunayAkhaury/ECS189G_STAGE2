# RNN.py
"""
RNN.py

Provides data loading, preprocessing, vocabulary building, dataset class, and RNN/LSTM/GRU model for text classification.
"""
import os
import re
import string
from collections import Counter
import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────────────────
# 1) Data loading and preprocessing
# ────────────────────────────────────────────────────────────────────────────────
def load_data(dataset_dir, split):
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
    Returns both word2idx and idx2word dictionaries.
    """
    counter = Counter()
    for text in texts:
        tokens = clean_text(text).split()
        counter.update(tokens)
    most_common = counter.most_common(max_vocab - 2)
    word2idx = {w: idx+2 for idx, (w, _) in enumerate(most_common)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


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
# 4) RNN/LSTM/GRU Model for Classification
# ────────────────────────────────────────────────────────────────────────────────
class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        rnn_units,
        num_layers=2,
        bidirectional=True,
        rnn_type='lstm',       # 'lstm', 'gru', or 'rnn'
        dropout=0.3,
        use_attention=False
    ):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers>1 else 0.0
        )
        if rnn_type=='gru':
            self.rnn = nn.GRU(**rnn_kwargs)
        elif rnn_type=='lstm':
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs)

        direction = 2 if bidirectional else 1
        self.fc_dropout = nn.Dropout(dropout)
        if use_attention:
            self.attn = nn.Linear(rnn_units * direction, 1)
        self.fc = nn.Linear(rnn_units * direction, 1)

    def forward(self, x):
        # x: [B, T]
        emb = self.embed_dropout(self.embedding(x))    # [B, T, E]
        out, _ = self.rnn(emb)                         # [B, T, H*dir]

        if self.use_attention:
            # attention scores over T
            scores  = self.attn(out).squeeze(-1)       # [B, T]
            weights = torch.softmax(scores, dim=1)     # [B, T]
            # weighted sum of hidden states
            h       = (out * weights.unsqueeze(-1)).sum(1)  # [B, H*dir]
        else:
            # just take the final time-step
            h = out[:, -1, :]                          # [B, H*dir]

        h = self.fc_dropout(h)
        logits = self.fc(h).squeeze(1)                 # [B]
        return logits

# ────────────────────────────────────────────────────────────────────────────────
# 5) Language Model Dataset and Model (FIXED INDENTATION)
# ────────────────────────────────────────────────────────────────────────────────
def build_lm_dataset(texts, word2idx, seq_len):
    """Build language modeling dataset for next-word prediction."""
    examples = []
    for t in texts:
        idxs = [word2idx.get(tok, 1) for tok in clean_text(t).split()]
        for i in range(len(idxs) - seq_len):
            examples.append((idxs[i:i + seq_len], idxs[i + seq_len]))
    X = torch.tensor([e[0] for e in examples], dtype=torch.long)
    y = torch.tensor([e[1] for e in examples], dtype=torch.long)
    return TensorDataset(X, y)


class RNNLanguageModel(nn.Module):
    """RNN Language Model for text generation."""
    def __init__(self, vocab_size, embed_dim, rnn_units,
                 num_layers=2, bidirectional=False,
                 rnn_type="lstm", dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        rnn_args = dict(
            input_size=embed_dim,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0)
        )
        if rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_args)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(**rnn_args)
        else:
            self.rnn = nn.LSTM(**rnn_args)

        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(rnn_units * factor, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        return self.fc(out), hidden  # [B,T,V], hidden

# ────────────────────────────────────────────────────────────────────────────────
# 6) Text Generation Utility
# ────────────────────────────────────────────────────────────────────────────────
def generate_text(model, word2idx, idx2word, seed_text,
                  gen_len=100, seq_len=5, temperature=1.0, device=None):
    """Generate text using a trained language model."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    tokens = clean_text(seed_text).split()
    gen = [word2idx.get(tok, 1) for tok in tokens]
    hidden = None
    for _ in range(gen_len):
        inp = torch.tensor([gen[-seq_len:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, hidden = model(inp, hidden)
        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        gen.append(nxt)
    return " ".join(idx2word.get(i, '<UNK>') for i in gen)