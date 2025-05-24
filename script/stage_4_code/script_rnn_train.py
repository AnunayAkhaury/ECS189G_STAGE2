# script_rnn_train.py
"""
Trains RNNClassifier on the dataset, evaluates, and saves model and training curves.
"""
import os
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from RNN import load_data, build_vocab, TextDataset, RNNClassifier

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
DATASET_DIR = '/Users/anunayakhaury/Personal/UCDavis/ECS189G/Project/ECS189G_STAGE2/data/stage_4_data/text_classification'
BATCH_SIZE  = 128
MAX_VOCAB   = 10000
MAX_LEN     = 200
EMBED_DIM   = 64
RNN_UNITS   = 64
NUM_LAYERS  = 1
BIDIR       = False
LR          = 1e-3
EPOCHS      = 10
MODEL_PATH  = 'rnn_model.pth'
HIST_PATH   = 'history.json'
CURVE_PATH  = 'learning_curves.png'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load data
train_texts, train_labels = load_data(DATASET_DIR, 'train')
test_texts, test_labels   = load_data(DATASET_DIR, 'test')

# 2) Build vocab and datasets
word2idx = build_vocab(train_texts, MAX_VOCAB)
vocab_size = len(word2idx)
train_ds = TextDataset(train_texts, train_labels, word2idx, MAX_LEN)
test_ds  = TextDataset(test_texts,  test_labels,  word2idx, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 3) Model, loss, optimizer
model = RNNClassifier(vocab_size, EMBED_DIM, RNN_UNITS, NUM_LAYERS, BIDIR).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4) Training loop
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
for epoch in range(1, EPOCHS+1):
    # Train
    model.train()
    total_loss = 0
    correct = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
    train_loss = total_loss / len(train_ds)
    train_acc  = correct / len(train_ds)

    # Evaluate
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()
    test_loss = total_loss / len(test_ds)
    test_acc  = correct / len(test_ds)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    print(f"Epoch {epoch}/{EPOCHS} "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Test loss: {test_loss:.4f}, acc: {test_acc:.4f}")

# 5) Save model & history
torch.save({'model_state_dict': model.state_dict(), 'word2idx': word2idx}, MODEL_PATH)
with open(HIST_PATH, 'w') as f:
    json.dump(history, f)
print(f"Model saved to {MODEL_PATH}")
print(f"History saved to {HIST_PATH}")

# 6) Plot learning curves
epochs = range(1, EPOCHS+1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, history['train_acc'], label='Train Acc')
plt.plot(epochs, history['test_acc'],  label='Test Acc')
plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs, history['train_loss'], label='Train Loss')
plt.plot(epochs, history['test_loss'],  label='Test Loss')
plt.title('Loss'); plt.legend()
plt.tight_layout()
plt.savefig(CURVE_PATH)
plt.show()
print(f"Learning curves saved to {CURVE_PATH}")