# script_rnn_train.py
"""
Trains RNNClassifier on the dataset with GloVe embeddings, evaluates, and saves model and training curves.
"""
import os, sys
import json
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

module_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # …/script/stage_4_code
        '..',  # …/script
        '..',  # …/ECS189G_STAGE2
        'local_code',
        'stage_4_code'
    )
)
sys.path.insert(0, module_path)

from RNN import load_data, setup_glove_embeddings, TextDataset, RNNClassifier


DATASET_DIR = '../../data/stage_4_data/text_classification'
GLOVE_PATH = '../../data/stage_4_data/glove.6B.100d.txt'
USE_GLOVE = True  # Set to False to use random embeddings instead
FREEZE_EMBEDDINGS = False  # Set to True to freeze GloVe embeddings during training

BATCH_SIZE = 128
MAX_VOCAB = 10000
MAX_LEN = 200
# EMBED_DIM will be determined by GloVe file (e.g., 100 for glove.6B.100d.txt)
RNN_UNITS = 256 #
NUM_LAYERS = 2 #
BIDIR = True
RNN_TYPE = 'rnn'  # 'lstm', 'gru', or 'rnn'
DROPOUT = 0.3 #
LR = 1e-3
EPOCHS = 10 #

# Output files
MODEL_PATH = 'rnn_model_glove.pth'
HIST_PATH = 'history_glove.json'
CURVE_PATH = 'learning_curves_glove.png'

# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():  # for Apple Silicon
    DEVICE = torch.device("mps")
    print("Using MPS")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

print(f"Configuration:")
print(f"  Dataset: {DATASET_DIR}")
print(f"  GloVe file: {GLOVE_PATH}")
print(f"  Use GloVe: {USE_GLOVE}")
print(f"  Freeze embeddings: {FREEZE_EMBEDDINGS}")
print(f"  Max vocab: {MAX_VOCAB}")
print(f"  Max length: {MAX_LEN}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LR}")
print(f"  Epochs: {EPOCHS}")
print()


print("Loading data...")
train_texts, train_labels = load_data(DATASET_DIR, 'train')
test_texts, test_labels = load_data(DATASET_DIR, 'test')
print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")


if USE_GLOVE and os.path.exists(GLOVE_PATH):
    print(f"\nSetting up vocabulary with GloVe embeddings...")
    word2idx, idx2word, embedding_matrix, embed_dim = setup_glove_embeddings(
        train_texts, GLOVE_PATH, MAX_VOCAB
    )
    pretrained_embeddings = embedding_matrix
else:
    if USE_GLOVE:
        print(f"Warning: GloVe file not found at {GLOVE_PATH}")
        print("Falling back to random embeddings...")
    else:
        print("Using random embeddings (GloVe disabled)...")

    # Fallback to original method
    from RNN import build_vocab

    word2idx, idx2word = build_vocab(train_texts, MAX_VOCAB)
    embed_dim = 128  # Default embedding dimension
    pretrained_embeddings = None

vocab_size = len(word2idx)
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embed_dim}")


print("\nCreating datasets...")
train_ds = TextDataset(train_texts, train_labels, word2idx, MAX_LEN)
test_ds = TextDataset(test_texts, test_labels, word2idx, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")


print(f"\nCreating model...")
model = RNNClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    rnn_units=RNN_UNITS,
    num_layers=NUM_LAYERS,
    bidirectional=BIDIR,
    rnn_type=RNN_TYPE,
    dropout=DROPOUT,
    pretrained_embeddings=pretrained_embeddings,
    freeze_embeddings=FREEZE_EMBEDDINGS
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


print(f"\nStarting training for {EPOCHS} epochs...")
print("=" * 80)

history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

for epoch in range(1, EPOCHS + 1):
    # Training phase
    model.train()
    total_loss = 0
    correct = 0
    num_batches = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        num_batches += 1

        # Print progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_loss = total_loss / len(train_ds)
    train_acc = correct / len(train_ds)

    model.eval()
    total_loss = 0
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()

            # collect for final metrics
            y_true.extend(y.int().cpu().tolist())
            y_pred.extend(preds.int().cpu().tolist())

    test_loss = total_loss / len(test_ds)
    test_acc = correct / len(test_ds)

    # record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    # print epoch summary
    print(f"Epoch {epoch}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    print("-" * 80)

print("Training completed!")
print("=" * 80)


print(f"\nSaving model and results...")

# Save model with all necessary information
model_save_dict = {
    'model_state_dict': model.state_dict(),
    'word2idx': word2idx,
    'idx2word': idx2word,
    'config': {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'rnn_units': RNN_UNITS,
        'num_layers': NUM_LAYERS,
        'bidirectional': BIDIR,
        'rnn_type': RNN_TYPE,
        'dropout': DROPOUT,
        'max_len': MAX_LEN,
        'used_glove': USE_GLOVE and os.path.exists(GLOVE_PATH),
        'frozen_embeddings': FREEZE_EMBEDDINGS
    },
    'final_performance': {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'test_loss': test_loss
    }
}

if pretrained_embeddings is not None:
    model_save_dict['embedding_matrix'] = embedding_matrix

torch.save(model_save_dict, MODEL_PATH)

# Save training history
with open(HIST_PATH, 'w') as f:
    json.dump(history, f, indent=2)

print(f"Model saved to: {MODEL_PATH}")
print(f"History saved to: {HIST_PATH}")


print(f"Creating learning curves...")

epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
plt.plot(epochs_range, history['test_acc'], 'r-', label='Test Accuracy', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
plt.plot(epochs_range, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Combined accuracy plot with final values
plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['train_acc'], 'b-', label=f'Train Acc (Final: {train_acc:.3f})', linewidth=2)
plt.plot(epochs_range, history['test_acc'], 'r-', label=f'Test Acc (Final: {test_acc:.3f})', linewidth=2)
plt.title('Final Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(CURVE_PATH, dpi=300, bbox_inches='tight')
plt.show()

print(f"Learning curves saved to: {CURVE_PATH}")

# ────────────────────────────────────────────────────────────────────────────────
# 8) Final summary
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"Dataset: {DATASET_DIR}")
print(f"GloVe embeddings: {'Yes' if (USE_GLOVE and os.path.exists(GLOVE_PATH)) else 'No'}")
if USE_GLOVE and os.path.exists(GLOVE_PATH):
    print(f"  - GloVe file: {GLOVE_PATH}")
    print(f"  - Embeddings frozen: {FREEZE_EMBEDDINGS}")
print(f"Vocabulary size: {vocab_size:,}")
print(f"Embedding dimension: {embed_dim}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()
print("FINAL PERFORMANCE:")
print(f"  Train Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"  Train Loss:     {train_loss:.4f}")
print(f"  Test Loss:      {test_loss:.4f}")
print()
print("FILES CREATED:")
print(f"  - Model: {MODEL_PATH}")
print(f"  - History: {HIST_PATH}")
print(f"  - Plots: {CURVE_PATH}")
print("=" * 80)

print("\nTo load and use this model later:")
print(f"""
# Load the saved model
checkpoint = torch.load('{MODEL_PATH}')
word2idx = checkpoint['word2idx']
config = checkpoint['config']

# Recreate the model
model = RNNClassifier(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for prediction
# text = "This movie was great!"
# ... (preprocessing and prediction code)
""")


y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X, y    = X.to(DEVICE), y.to(DEVICE)
        logits  = model(X)
        preds   = (torch.sigmoid(logits) >= 0.5).float()
        y_true.extend(y.int().cpu().tolist())
        y_pred.extend(preds.int().cpu().tolist())

metrics = {}

# overall accuracy
metrics['accuracy'] = accuracy_score(y_true, y_pred)

# weighted precision, recall, f1
p_w, r_w, f1_w, _ = precision_recall_fscore_support(
    y_true, y_pred, average='weighted'
)
metrics['weighted'] = {
    'precision': p_w,
    'recall':    r_w,
    'f1':        f1_w
}

# macro-averaged
p_m, r_m, f1_m, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro'
)
metrics['macro'] = {
    'precision': p_m,
    'recall':    r_m,
    'f1':        f1_m
}

# micro-averaged
p_i, r_i, f1_i, _ = precision_recall_fscore_support(
    y_true, y_pred, average='micro'
)
metrics['micro'] = {
    'precision': p_i,
    'recall':    r_i,
    'f1':        f1_i
}

print('************ Overall Performance ************')
print(f"Accuracy: {metrics['accuracy']:.4f}")

print('\nWeighted Metrics:')
print(f"Precision: {metrics['weighted']['precision']:.4f}")
print(f"Recall:    {metrics['weighted']['recall']:.4f}")
print(f"F1-score:  {metrics['weighted']['f1']:.4f}")

print('\nMacro Metrics:')
print(f"Precision: {metrics['macro']['precision']:.4f}")
print(f"Recall:    {metrics['macro']['recall']:.4f}")
print(f"F1-score:  {metrics['macro']['f1']:.4f}")

print('\nMicro Metrics:')
print(f"Precision: {metrics['micro']['precision']:.4f}")
print(f"Recall:    {metrics['micro']['recall']:.4f}")
print(f"F1-score:  {metrics['micro']['f1']:.4f}")

print('************ Finish ************')
