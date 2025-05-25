# script_rnn_generate_with_glove.py
"""
Trains an RNNLanguageModel on the text generation dataset and generates a story.
Updated to support GloVe embeddings.
"""
import os
import sys
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
print(">>> MODULE_PATH =", module_path)
print(">>> FILES IN MODULE_PATH =", os.listdir(module_path))

from RNN import build_vocab, build_lm_dataset, RNNLanguageModel, generate_text, setup_glove_embeddings


def main():
    parser = argparse.ArgumentParser(description="Train RNN LM and generate text")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing .txt files for language modeling")
    parser.add_argument('--glove_path', type=str, default=None,
                        help="Path to GloVe embeddings file (e.g., glove.6B.100d.txt)")
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help="Freeze pre-trained embeddings during training")
    parser.add_argument('--max_vocab', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=5,
                        help="Input sequence length for next-word prediction")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--rnn_units', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed_text', type=str, required=True,
                        help="Seed text (e.g. three words) to start generation")
    parser.add_argument('--gen_len', type=int, default=100,
                        help="Number of words to generate")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Temperature for text generation (higher = more random)")
    args = parser.parse_args()

    # device selection
    if torch.cuda.is_available():
        device = torch.device('cuda');
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps');
        print("Using MPS")
    else:
        device = torch.device('cpu');
        print("Using CPU")

    # load raw texts
    texts = []
    for fname in os.listdir(args.data_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(args.data_dir, fname), encoding='utf-8') as f:
                texts.append(f.read())

    print(f"Loaded {len(texts)} text files")

    # Setup vocabulary and embeddings
    if args.glove_path and os.path.exists(args.glove_path):
        print("Setting up GloVe embeddings...")
        word2idx, idx2word, embedding_matrix, embed_dim = setup_glove_embeddings(
            texts, args.glove_path, args.max_vocab
        )
        # Override embed_dim from GloVe
        args.embed_dim = embed_dim
        pretrained_embeddings = embedding_matrix
        print(f"Using GloVe embeddings with dimension: {embed_dim}")
    else:
        if args.glove_path:
            print(f"Warning: GloVe file not found at {args.glove_path}, using random embeddings")
        print("Using random embeddings...")
        word2idx, idx2word = build_vocab(texts, args.max_vocab)
        pretrained_embeddings = None

    vocab_size = len(word2idx)

    # Add this debug code to your generation script after loading the model:

    print(f"\n=== DEBUGGING UNK TOKENS ===")
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"UNK token index: {word2idx.get('<UNK>', 'NOT FOUND')}")

    # Check if your seed text tokens are in vocabulary
    from RNN import clean_text
    seed_cleaned = clean_text("what do you call a", mode="generation")
    print(f"Seed text cleaned: '{seed_cleaned}'")
    seed_tokens = seed_cleaned.split()
    print(f"Seed tokens: {seed_tokens}")

    for token in seed_tokens:
        idx = word2idx.get(token, -1)
        if idx == -1:
            print(f"  ❌ '{token}' -> NOT IN VOCAB!")
        elif idx == 1:  # UNK index
            print(f"  ⚠️  '{token}' -> UNK (index 1)")
        else:
            print(f"  ✅ '{token}' -> {idx}")

    # Check what word index 1 maps to
    print(f"\nIndex 1 maps to: '{idx2word.get(1, 'MISSING')}'")

    # Show some high-index words (these might be getting generated)
    max_idx = len(word2idx) - 1
    print(f"Max vocabulary index: {max_idx}")
    print(f"Last few words in vocab:")
    for i in range(max(0, max_idx - 5), max_idx + 1):
        print(f"  {i}: '{idx2word.get(i, 'MISSING')}'")

    print(f"Vocabulary size: {vocab_size}")

    # build language modeling dataset
    lm_dataset = build_lm_dataset(texts, word2idx, args.seq_len)
    train_loader = DataLoader(lm_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Training samples: {len(lm_dataset)}")

    # model, loss, optimizer
    model = RNNLanguageModel(
        vocab_size,
        args.embed_dim,
        args.rnn_units,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=args.freeze_embeddings
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("\nStarting training...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        num_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(X)
            # predict next word at final time-step
            loss = criterion(logits[:, -1, :], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            num_batches += 1

        avg_loss = total_loss / len(lm_dataset)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    # generate text
    print(f"\nGenerating text with seed: '{args.seed_text}'")
    generated = generate_text(
        model,
        word2idx,
        idx2word,
        seed_text=args.seed_text,
        gen_len=args.gen_len,
        seq_len=args.seq_len,
        temperature=args.temperature,
        device=device
    )
    print("\n" + "=" * 50)
    print("GENERATED STORY")
    print("=" * 50)
    print(generated)
    print("=" * 50)

    # compare with a random training sample
    print("\n" + "=" * 50)
    print("SAMPLE FROM TRAINING DATA")
    print("=" * 50)
    sample = random.choice(texts)
    print(sample[:min(len(sample), 300)] + "..." if len(sample) > 300 else sample)
    print("=" * 50)

    # save model and vocab
    save_dict = {
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'args': vars(args),
        'vocab_size': vocab_size
    }
    torch.save(save_dict, 'lm_model.pth')

    with open('lm_vocab.json', 'w') as f:
        json.dump({
            'word2idx': word2idx,
            'idx2word': idx2word,
            'vocab_size': vocab_size,
            'embed_dim': args.embed_dim
        }, f, indent=2)

    print(f"\nModel and vocabulary saved to:")
    print(f"  - lm_model.pth")
    print(f"  - lm_vocab.json")


if __name__ == '__main__':
    main()