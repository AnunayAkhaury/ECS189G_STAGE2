# script_rnn_generate.py
"""
Trains an RNNLanguageModel on the text generation dataset and generates a story.
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
        os.path.dirname(__file__),   # …/script/stage_4_code
        '..',                         # …/script
        '..',                         # …/ECS189G_STAGE2
        'local_code',
        'stage_4_code'
    )
)
sys.path.insert(0, module_path)
print(">>> MODULE_PATH =", module_path)
print(">>> FILES IN MODULE_PATH =", os.listdir(module_path))

from RNN import build_vocab, build_lm_dataset, RNNLanguageModel, generate_text


def main():
    parser = argparse.ArgumentParser(description="Train RNN LM and generate text")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Directory containing .txt files for language modeling")
    parser.add_argument('--max_vocab', type=int, default=10000)
    parser.add_argument('--seq_len', type=int, default=5,
                        help="Input sequence length for next-word prediction")
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--rnn_units', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        choices=['rnn','lstm','gru'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed_text', type=str, required=True,
                        help="Seed text (e.g. three words) to start generation")
    parser.add_argument('--gen_len', type=int, default=100,
                        help="Number of words to generate")
    args = parser.parse_args()

    # device selection
    if torch.cuda.is_available():
        device = torch.device('cuda'); print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps');  print("Using MPS")
    else:
        device = torch.device('cpu'); print("Using CPU")

    # load raw texts
    texts = []
    for fname in os.listdir(args.data_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(args.data_dir, fname), encoding='utf-8') as f:
                texts.append(f.read())

    # build vocab
    word2idx, idx2word = build_vocab(texts, args.max_vocab)
    vocab_size = len(word2idx)

    # build language modeling dataset
    lm_dataset = build_lm_dataset(texts, word2idx, args.seq_len)
    train_loader = DataLoader(lm_dataset, batch_size=args.batch_size, shuffle=True)

    # model, loss, optimizer
    model = RNNLanguageModel(
        vocab_size,
        args.embed_dim,
        args.rnn_units,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        rnn_type=args.rnn_type,
        dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
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
        avg_loss = total_loss / len(lm_dataset)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

    # generate text
    generated = generate_text(
        model,
        word2idx,
        idx2word,
        seed_text=args.seed_text,
        gen_len=args.gen_len,
        seq_len=args.seq_len,
        device=device
    )
    print("\n=== Generated Story ===\n")
    print(generated)

    # compare with a random training sample
    print("\n=== Sample from Training Data ===\n")
    sample = random.choice(texts)
    print(sample[:min(len(sample), 200)])

    # save model and vocab
    torch.save({'model_state_dict': model.state_dict(), 'word2idx': word2idx}, 'lm_model.pth')
    with open('lm_vocab.json', 'w') as f:
        json.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    print("\nModel and vocabulary saved.")


if __name__ == '__main__':
    main()
