#!/usr/bin/env python3
# train_classifier.py
"""
Command-line training script for RNN text classification with GloVe embeddings.
Supports flexible configuration via command-line arguments.
"""
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# Add the RNN module to path
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

from RNN import load_data, setup_glove_embeddings, build_vocab, TextDataset, RNNClassifier


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RNN classifier for text classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='../../data/stage_4_data/text_classification',
                        help='Path to dataset directory')
    parser.add_argument('--glove_path', type=str,
                        default='../../data/stage_4_data/glove.6B.100d.txt',
                        help='Path to GloVe embeddings file')
    parser.add_argument('--use_glove', action='store_true', default=True,
                        help='Use GloVe embeddings (default: True)')
    parser.add_argument('--no_glove', dest='use_glove', action='store_false',
                        help='Disable GloVe embeddings')
    parser.add_argument('--freeze_embeddings', action='store_true', default=False,
                        help='Freeze embedding weights during training')

    # Model architecture
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Maximum vocabulary size')
    parser.add_argument('--max_len', type=int, default=200,
                        help='Maximum sequence length')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Embedding dimension (auto-detected from GloVe if not specified)')
    parser.add_argument('--rnn_units', type=int, default=128,
                        help='Number of RNN hidden units')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'rnn'],
                        help='Type of RNN cell')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Use bidirectional RNN (default: True)')
    parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false',
                        help='Disable bidirectional RNN')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention mechanism (default: True)')
    parser.add_argument('--no_attention', dest='use_attention', action='store_false',
                        help='Disable attention mechanism')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for regularization')

    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for output files')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save training plots (default: True)')
    parser.add_argument('--no_plots', dest='save_plots', action='store_false',
                        help='Disable saving plots')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval for training batches')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using specified device: {device}")

    return device


def setup_optimizer(model, optimizer_name, lr, weight_decay):
    """Setup optimizer based on name."""
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_experiment_name(args):
    """Create experiment name if not provided."""
    if args.experiment_name:
        return args.experiment_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    glove_str = "glove" if args.use_glove else "random"
    attention_str = "attn" if args.use_attention else "no_attn"
    bidir_str = "bidir" if args.bidirectional else "unidir"

    return f"{args.rnn_type}_{glove_str}_{attention_str}_{bidir_str}_{timestamp}"


def train_epoch(model, train_loader, criterion, optimizer, device, log_interval):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()

        if batch_idx % log_interval == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss, correct


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == y).sum().item()

    return total_loss, correct


def save_results(args, model, word2idx, idx2word, history, embedding_matrix, output_dir, experiment_name):
    """Save model, history, and configuration."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f"{experiment_name}_model.pth")
    model_save_dict = {
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'args': vars(args),
        'config': {
            'vocab_size': len(word2idx),
            'embed_dim': args.embed_dim,
            'rnn_units': args.rnn_units,
            'num_layers': args.num_layers,
            'bidirectional': args.bidirectional,
            'rnn_type': args.rnn_type,
            'dropout': args.dropout,
            'use_attention': args.use_attention,
            'max_len': args.max_len,
        },
        'final_performance': {
            'train_acc': history['train_acc'][-1],
            'test_acc': history['test_acc'][-1],
            'train_loss': history['train_loss'][-1],
            'test_loss': history['test_loss'][-1]
        }
    }

    if embedding_matrix is not None:
        model_save_dict['embedding_matrix'] = embedding_matrix

    torch.save(model_save_dict, model_path)

    # Save history
    history_path = os.path.join(output_dir, f"{experiment_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save configuration
    config_path = os.path.join(output_dir, f"{experiment_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    return model_path, history_path, config_path


def plot_results(history, output_dir, experiment_name, save_plots):
    """Plot and optionally save training curves."""
    epochs_range = range(1, len(history['train_acc']) + 1)

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

    # Combined performance
    plt.subplot(1, 3, 3)
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    plt.plot(epochs_range, history['train_acc'], 'b-',
             label=f'Train Acc (Final: {final_train_acc:.3f})', linewidth=2)
    plt.plot(epochs_range, history['test_acc'], 'r-',
             label=f'Test Acc (Final: {final_test_acc:.3f})', linewidth=2)
    plt.title('Final Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plot_path = os.path.join(output_dir, f"{experiment_name}_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_path}")

    plt.show()


def main():
    """Main training function."""
    args = parse_arguments()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Setup device
    device = setup_device(args.device)

    # Create experiment name
    experiment_name = create_experiment_name(args)

    print("=" * 80)
    print("RNN TEXT CLASSIFICATION TRAINING")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print()

    # Load data
    print("Loading data...")
    train_texts, train_labels = load_data(args.data_dir, 'train')
    test_texts, test_labels = load_data(args.data_dir, 'test')
    print(f"Loaded {len(train_texts)} training and {len(test_texts)} test samples")

    # Setup vocabulary and embeddings
    if args.use_glove and os.path.exists(args.glove_path):
        print(f"\nSetting up GloVe embeddings from {args.glove_path}...")
        word2idx, idx2word, embedding_matrix, embed_dim = setup_glove_embeddings(
            train_texts, args.glove_path, args.vocab_size
        )
        pretrained_embeddings = embedding_matrix
        args.embed_dim = embed_dim  # Update embed_dim from GloVe
    else:
        if args.use_glove:
            print(f"Warning: GloVe file not found at {args.glove_path}")
            print("Falling back to random embeddings...")

        word2idx, idx2word = build_vocab(train_texts, args.vocab_size)
        embedding_matrix = None
        pretrained_embeddings = None
        if args.embed_dim is None:
            args.embed_dim = 128  # Default embedding dimension

    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {args.embed_dim}")

    # Create datasets
    print("\nCreating datasets...")
    train_ds = TextDataset(train_texts, train_labels, word2idx, args.max_len)
    test_ds = TextDataset(test_texts, test_labels, word2idx, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Create model
    print(f"\nCreating {args.rnn_type.upper()} model...")
    model = RNNClassifier(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        use_attention=args.use_attention,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=args.freeze_embeddings
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = setup_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        total_loss, correct = train_epoch(model, train_loader, criterion, optimizer,
                                          device, args.log_interval)
        train_loss = total_loss / len(train_ds)
        train_acc = correct / len(train_ds)

        # Evaluate
        total_loss, correct = evaluate(model, test_loader, criterion, device)
        test_loss = total_loss / len(test_ds)
        test_acc = correct / len(test_ds)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Acc: {test_acc:.4f}")
        print("-" * 80)

    # Save results
    print("\nSaving results...")
    model_path, history_path, config_path = save_results(
        args, model, word2idx, idx2word, history, embedding_matrix,
        args.output_dir, experiment_name
    )

    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print(f"Config saved to: {config_path}")

    # Plot results
    plot_results(history, args.output_dir, experiment_name, args.save_plots)

    # Final summary
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Experiment: {experiment_name}")
    print(f"Final Train Accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"Final Test Accuracy:  {final_test_acc:.4f} ({final_test_acc * 100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()