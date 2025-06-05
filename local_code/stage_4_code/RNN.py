
import os
import re
import string
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torch.nn as nn

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


def clean_text(text, mode="classification"):
    if mode == "classification":
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', ' ', text)  # Remove URLs
        text = re.sub(r'www\.\S+', ' ', text)  # Remove www links
        text = text.translate(str.maketrans('', '', string.punctuation))
    elif mode == "generation":
        # Keep only question marks and remove all other punctuation
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)
        text = re.sub(r'["""''`]', '', text)
        text = re.sub(r'\([^)]*\)', ' ', text)  # Remove (parenthetical content)
        text = re.sub(r'\[[^\]]*\]', ' ', text)  # Remove [bracketed content]
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.!,:;]', ' ', text)  # Remove periods, exclamations, commas, etc.
        text = re.sub(r'([?])', r' \1 ', text)
        text = re.sub(r'[~@#$%^&*+=\[\]{}\\|<>_!.\-]', ' ', text)
        #strip whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    return text


def load_glove_embeddings(glove_path):
    """
    Load GloVe embeddings from file.
    Returns dictionary mapping words to embedding vectors.
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    embeddings_dict = {}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_dict[word] = vector

    embed_dim = len(next(iter(embeddings_dict.values())))
    print(f"Loaded {len(embeddings_dict)} word vectors of dimension {embed_dim}")
    return embeddings_dict, embed_dim


def create_embedding_matrix(word2idx, glove_embeddings, embed_dim):
    """
    Create embedding matrix from GloVe embeddings for words in vocabulary.
    Words not found in GloVe will have random embeddings.
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim))

    # Set padding token to zeros
    embedding_matrix[0] = np.zeros(embed_dim)

    found_words = 0
    for word, idx in word2idx.items():
        if word in glove_embeddings:
            embedding_matrix[idx] = glove_embeddings[word]
            found_words += 1

    print(f"Found {found_words}/{vocab_size} words in GloVe embeddings")
    return torch.tensor(embedding_matrix, dtype=torch.float32)



def build_vocab(texts, max_vocab, glove_embeddings=None, mode="classification"):
    """
    Build a word2idx mapping of the most common max_vocab-2 words, with 0 for PAD and 1 for UNK.
    If glove_embeddings provided, prioritize words that exist in GloVe.
    Returns both word2idx and idx2word dictionaries.
    """
    counter = Counter()
    for text in texts:
        tokens = clean_text(text, mode=mode).split()
        counter.update(tokens)

    # If using GloVe, prioritize words that exist in GloVe embeddings
    if glove_embeddings is not None:
        # Separate words into those in GloVe and those not in GloVe
        glove_words = []
        non_glove_words = []

        for word, count in counter.most_common():
            if word in glove_embeddings:
                glove_words.append((word, count))
            else:
                non_glove_words.append((word, count))

        # Take most common words, prioritizing GloVe words
        selected_words = []
        remaining_slots = max_vocab - 3  # Reserve slots for PAD and UNK and ?

        # First add GloVe words
        for word, count in glove_words:
            if len(selected_words) < remaining_slots:
                selected_words.append(word)

        # Then add non-GloVe words if there's space
        for word, count in non_glove_words:
            if len(selected_words) < remaining_slots:
                selected_words.append(word)

        print(
            f"Selected {len([w for w in selected_words if w in glove_embeddings])} GloVe words out of {len(selected_words)} total words")
    else:
        most_common = counter.most_common(max_vocab - 3)
        selected_words = [word for word, _ in most_common]

    # Create mappings
    word2idx = {w: idx + 3 for idx, w in enumerate(selected_words)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    word2idx['?'] = 2
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def encode_texts(texts, word2idx, max_len, mode="classification"):
    """
    Convert list of raw texts to a tensor of shape (len(texts), max_len) of indices.
    """
    sequences = []
    for text in texts:
        tokens = clean_text(text, mode=mode).split()
        idxs = [word2idx.get(tok, 1) for tok in tokens]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        sequences.append(idxs)
    return torch.tensor(sequences, dtype=torch.long)



class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len, mode="classification"):
        self.X = encode_texts(texts, word2idx, max_len, mode=mode)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class RNNClassifier(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            rnn_units,
            num_layers=2,
            bidirectional=True,
            rnn_type='lstm',  # 'lstm', 'gru', or 'rnn'
            dropout=0.3,
            pretrained_embeddings=None,
            freeze_embeddings=False
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Load pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            print("Loading pre-trained embeddings...")
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                print("Embeddings frozen (will not be updated during training)")
            else:
                print("Embeddings will be fine-tuned during training")

        self.embed_dropout = nn.Dropout(dropout)

        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        if rnn_type == 'gru':
            self.rnn = nn.GRU(**rnn_kwargs)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.RNN(**rnn_kwargs)

        direction = 2 if bidirectional else 1
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_units * direction, 1)

    def forward(self, x):
        emb = self.embed_dropout(self.embedding(x))  # [B, T, E]
        out, _ = self.rnn(emb)  # [B, T, H*dir]

        if isinstance(self.rnn, nn.RNN):
            h = out.max(dim=1)[0]  # [B, H*dir]
        else:
            h = out[:, -1, :]  # [B, H*dir]

        h = self.fc_dropout(h)
        logits = self.fc(h).squeeze(1)  # [B]
        return logits



def build_lm_dataset(texts, word2idx, seq_len):
    """Build language modeling dataset for next-word prediction."""
    examples = []
    for t in texts:
        idxs = [word2idx.get(tok, 1) for tok in clean_text(t, mode="generation").split()]
        for i in range(len(idxs) - seq_len):
            examples.append((idxs[i:i + seq_len], idxs[i + seq_len]))
    X = torch.tensor([e[0] for e in examples], dtype=torch.long)
    y = torch.tensor([e[1] for e in examples], dtype=torch.long)
    return TensorDataset(X, y)


def build_vocab_for_lm(texts, max_vocab, glove_embeddings=None):
    """
    Build vocabulary specifically for language modeling (uses generation mode).
    """
    return build_vocab(texts, max_vocab, glove_embeddings, mode="generation")


def setup_glove_embeddings_for_lm(texts, glove_path, max_vocab):
    """
    Convenience function to set up vocabulary and embeddings with GloVe for language modeling.
    Always uses generation mode.
    """
    # Load GloVe embeddings
    glove_embeddings, embed_dim = load_glove_embeddings(glove_path)

    # Build vocabulary with GloVe prioritization using generation mode
    word2idx, idx2word = build_vocab(texts, max_vocab, glove_embeddings, mode="generation")

    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(word2idx, glove_embeddings, embed_dim)

    return word2idx, idx2word, embedding_matrix, embed_dim


class RNNLanguageModel(nn.Module):
    """RNN Language Model for text generation."""

    def __init__(self, vocab_size, embed_dim, rnn_units,
                 num_layers=2, bidirectional=False,
                 rnn_type="lstm", dropout=0.3,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        #Added line
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Load pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

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



def generate_text(model, word2idx, idx2word, seed_text,
                  gen_len=100, seq_len=5, temperature=1.0, device=None):
    """Generate text using a trained language model."""
    print(word2idx.keys())
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    tokens = clean_text(seed_text, mode="generation").split()
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
    print(gen)
    return " ".join(idx2word.get(i, '') for i in gen)



def setup_glove_embeddings(texts, glove_path, max_vocab, mode="classification"):
    """
    Convenience function to set up vocabulary and embeddings with GloVe.

    Args:
        texts: List of training texts
        glove_path: Path to GloVe embeddings file (e.g., 'glove.6B.100d.txt')
        max_vocab: Maximum vocabulary size
        mode: "classification" or "generation" for different text cleaning

    Returns:
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        embedding_matrix: Pre-trained embedding matrix
        embed_dim: Embedding dimension
    """
    # Load GloVe embeddings
    glove_embeddings, embed_dim = load_glove_embeddings(glove_path)

    # Build vocabulary with GloVe prioritization
    word2idx, idx2word = build_vocab(texts, max_vocab, glove_embeddings, mode=mode)

    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(word2idx, glove_embeddings, embed_dim)

    return word2idx, idx2word, embedding_matrix, embed_dim
