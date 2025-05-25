# script_rnn_generate_bert.py
"""
Enhanced RNN text generation with BERT tokenizer and embeddings
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Add your module path
module_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        'local_code',
        'stage_4_code'
    )
)
sys.path.insert(0, module_path)

# Import your BERT-enhanced functions
from RNN_with_BERT import (
    BERTEnhancedRNN,
    build_bert_lm_dataset,
    generate_text_bert,
    train_bert_enhanced_model
)


def main():
    parser = argparse.ArgumentParser(description="Train BERT-enhanced RNN and generate text")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help="BERT model to use")
    parser.add_argument('--freeze_bert', action='store_true',
                        help="Freeze BERT weights during training")
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--rnn_units', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--rnn_type', type=str, default='lstm')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)  # Lower LR for BERT
    parser.add_argument('--seed_text', type=str, required=True)
    parser.add_argument('--gen_len', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load BERT tokenizer
    print(f"Loading BERT tokenizer: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # Load texts
    texts = []
    for fname in os.listdir(args.data_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(args.data_dir, fname), encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty texts
                    texts.append(content)

    print(f"Loaded {len(texts)} text files")

    # Build dataset with BERT tokenizer
    print("Building dataset with BERT tokenizer...")
    lm_dataset = build_bert_lm_dataset(texts, tokenizer, args.seq_len)

    if lm_dataset is None or len(lm_dataset) == 0:
        print("ERROR: No training examples created!")
        return

    print(f"Created {len(lm_dataset)} training examples")

    # Create data loader
    train_loader = DataLoader(lm_dataset, batch_size=args.batch_size, shuffle=True)

    # Create BERT-enhanced model
    print("Creating BERT-enhanced RNN model...")
    model = BERTEnhancedRNN(
        bert_model_name=args.bert_model,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert
    )

    print(f"Model created. BERT frozen: {args.freeze_bert}")

    # Train the model
    print("Starting training...")
    model = train_bert_enhanced_model(
        model, train_loader, args.epochs, args.lr, device
    )

    # Generate text
    print(f"\n=== Generating text with seed: '{args.seed_text}' ===")
    generated = generate_text_bert(
        model=model,
        tokenizer=tokenizer,
        seed_text=args.seed_text,
        gen_len=args.gen_len,
        seq_len=args.seq_len,
        temperature=args.temperature,
        device=device
    )

    print(f"Generated: {generated}")

    # Generate a few more examples with different temperatures
    print("\n=== Additional generations ===")
    for temp in [0.7, 1.0, 1.3]:
        gen = generate_text_bert(
            model, tokenizer, args.seed_text,
            gen_len=min(args.gen_len, 30), temperature=temp, device=device
        )
        print(f"Temp {temp}: {gen}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_name': args.bert_model,
        'model_config': {
            'rnn_units': args.rnn_units,
            'num_layers': args.num_layers,
            'rnn_type': args.rnn_type,
            'dropout': args.dropout,
            'freeze_bert': args.freeze_bert
        }
    }, 'bert_enhanced_model.pth')

    print("Model saved as 'bert_enhanced_model.pth'")


if __name__ == '__main__':
    main()