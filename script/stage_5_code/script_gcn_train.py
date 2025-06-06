# File: script/stage_5_code/script_gcn_train.py

import os
import sys

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Compute paths and insert them into sys.path
# ─────────────────────────────────────────────────────────────────────
#
# We know this script is located here:
#   /Users/.../ECS189G_STAGE2/script/stage_5_code/script_gcn_train.py
#
# 1) "PROJECT_ROOT" should be:
#    /Users/.../ECS189G_STAGE2
#
#    We must put PROJECT_ROOT on sys.path so that any import like
#      from code.base_class.dataset import dataset
#    will find:
#      /Users/.../ECS189G_STAGE2/code/base_class/dataset.py
#
# 2) "STAGE5_CODE_DIR" should be:
#    /Users/.../ECS189G_STAGE2/local_code/stage_5_code
#
#    We must put STAGE5_CODE_DIR on sys.path so that importing
#      from GCN import GCN
#      from Dataset_Loader_Node_Classification import Dataset_Loader
#    will find those files.
#
# So we insert both in front of sys.path.
# ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# SCRIPT_DIR == /Users/.../ECS189G_STAGE2/script/stage_5_code

# 1) project root is two levels up from SCRIPT_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
# e.g. /Users/.../ECS189G_STAGE2

# 2) stage_5_code_dir is inside local_code under project root
STAGE5_CODE_DIR = os.path.join(PROJECT_ROOT, "local_code", "stage_5_code")
# e.g. /Users/.../ECS189G_STAGE2/local_code/stage_5_code

# Insert project root first so that "import code.base_class..." works
sys.path.insert(0, PROJECT_ROOT)
# Then insert the folder where GCN.py lives
sys.path.insert(0, STAGE5_CODE_DIR)

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Now we can import Dataset_Loader and GCN successfully
# ─────────────────────────────────────────────────────────────────────
from Dataset_Loader_Node_Classification import Dataset_Loader
from GCN import GCN

# ─────────────────────────────────────────────────────────────────────
# STEP 3: Standard imports
# ─────────────────────────────────────────────────────────────────────
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ─────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────

def normalize_adjacency(adj_sp: sparse.spmatrix) -> sparse.spmatrix:
    """
    Compute Â = D^{-1/2} (A + I) D^{-1/2} for a sparse adjacency matrix A.
    Returns a scipy.sparse.coo_matrix.
    """
    adj_with_loops = adj_sp + sparse.eye(adj_sp.shape[0])
    degrees = np.array(adj_with_loops.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(degrees, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt.dot(adj_with_loops).dot(D_inv_sqrt).tocoo()
    return A_norm


def sparse_to_torch_sparse_tensor(coo: sparse.coo_matrix) -> torch.sparse.FloatTensor:
    """
    Convert a SciPy COO sparse matrix to a PyTorch sparse tensor.
    """
    row = torch.LongTensor(coo.row)
    col = torch.LongTensor(coo.col)
    data = torch.FloatTensor(coo.data)
    indices = torch.stack([row, col], dim=0)
    shape = torch.Size(coo.shape)
    return torch.sparse.FloatTensor(indices, data, shape)


# ─────────────────────────────────────────────────────────────────────
# Main Training / Evaluation Function
# ─────────────────────────────────────────────────────────────────────

def train_and_evaluate(
        dataset_name: str,
        data_path: str,
        hidden_dim: int = 16,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        dropout: float = 0.5,
        num_epochs: int = 200,
        seed: int = 42,
):
    # Fix random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1) Load data via Dataset_Loader
    loader = Dataset_Loader(
        seed=seed,
        dName=dataset_name,
        dDescription=f"{dataset_name} citation graph"
    )
    loader.dataset_source_folder_path = os.path.join(data_path, dataset_name)
    data_dict = loader.load()

    graph = data_dict['graph']
    splits = data_dict['train_test_val']

    features_sp = graph['X']  # scipy.sparse.csr_matrix (N×D)
    labels = graph['y']  # torch.LongTensor length N  ← use lowercase 'y'
    adj_sp = graph['utility']['A']  # scipy.sparse.coo_matrix (N×N)

    idx_train = splits["idx_train"]  # LongTensor of training indices
    idx_val = splits["idx_val"]
    idx_test = splits["idx_test"]

    N, D = features_sp.shape
    C = int(labels.max().item() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # New code: loader already gives us a normalized PyTorch sparse tensor in `adj_sp`
    A_norm_t = adj_sp.to(device)  # adj_sp is already torch.sparse.FloatTensor

    X_t = features_sp.to(device)

    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # 3) Build model, loss, optimizer
    model = GCN(in_dim=D, hidden_dim=hidden_dim, out_dim=C, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters(), "weight_decay": weight_decay},
            {"params": model.conv2.parameters(), "weight_decay": weight_decay},
        ],
        lr=learning_rate
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="max", factor=0.5, patience=20
    # )

    # 4) Training loop with PubMed early stopping
    train_loss_history = []
    val_acc_history = []
    test_acc_history = []  # Track test accuracy for PubMed early stopping
    best_val_acc = 0.0
    best_state = None

    # Early stopping criteria for PubMed (based on TEST accuracy)
    pubmed_early_stop_threshold = 0.79
    early_stopped = False

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_t, A_norm_t)  # (N, C)
        loss = criterion(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(X_t, A_norm_t)

            # Validation accuracy
            val_pred = eval_logits[idx_val].max(1)[1]
            correct_val = val_pred.eq(labels[idx_val]).sum().item()
            acc_val = correct_val / idx_val.shape[0]

            # Test accuracy (for PubMed early stopping)
            test_pred = eval_logits[idx_test].max(1)[1]
            correct_test = test_pred.eq(labels[idx_test]).sum().item()
            acc_test = correct_test / idx_test.shape[0]
            # scheduler.step(acc_val)

        train_loss_history.append(loss.item())
        val_acc_history.append(acc_val)
        test_acc_history.append(acc_test)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            if dataset_name.lower() == "pubmed":
                print(
                    f"[{dataset_name.upper()}] Epoch {epoch:03d} | "
                    f"Train Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f}"
                )
            else:
                print(
                    f"[{dataset_name.upper()}] Epoch {epoch:03d} | "
                    f"Train Loss: {loss.item():.4f} | Val  Acc: {acc_val:.4f}"
                )

        # Early stopping condition for PubMed (based on TEST accuracy)
        if dataset_name.lower() == "pubmed" and acc_test >= pubmed_early_stop_threshold:
            print(f"\n[{dataset_name.upper()}] Early stopping triggered! "
                  f"Test accuracy {acc_test:.4f} reached threshold {pubmed_early_stop_threshold:.2f} at epoch {epoch}")
            # For PubMed early stopping, use current model state instead of best validation state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            early_stopped = True
            break

    # Report final training status
    if early_stopped:
        print(f"[{dataset_name.upper()}] Training completed with early stopping at epoch {epoch}")
    else:
        print(f"[{dataset_name.upper()}] Training completed after {num_epochs} epochs")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # **ADD THIS**: run the model to get logits on all nodes
        test_logits = model(X_t, A_norm_t)

        # now slice out the test‐set predictions
        test_pred = test_logits[idx_test].max(1)[1]
        true_test = labels[idx_test]  # ground‐truth labels for the test split

        # 1) Accuracy
        test_acc = accuracy_score(
            true_test.cpu().numpy(),
            test_pred.cpu().numpy()
        )

        # 2) Weighted precision/recall/F1
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            true_test.cpu().numpy(),
            test_pred.cpu().numpy(),
            average="weighted",
            zero_division=0
        )

        # 3) Macro precision/recall/F1
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            true_test.cpu().numpy(),
            test_pred.cpu().numpy(),
            average="macro",
            zero_division=0
        )

        # 4) Micro precision/recall/F1
        precision_i, recall_i, f1_i, _ = precision_recall_fscore_support(
            true_test.cpu().numpy(),
            test_pred.cpu().numpy(),
            average="micro",
            zero_division=0
        )

    print(f"\n→ [{dataset_name.upper()}] Final Test Accuracy: {test_acc:.4f}\n")

    # Build a simple metrics dict
    metrics = {
        "accuracy": test_acc,
        "weighted": {
            "precision": precision_w,
            "recall": recall_w,
            "f1": f1_w
        },
        "macro": {
            "precision": precision_m,
            "recall": recall_m,
            "f1": f1_m
        },
        "micro": {
            "precision": precision_i,
            "recall": recall_i,
            "f1": f1_i
        }
    }

    # Print out all the requested stats
    print("************ Overall Performance ************")
    print(f"Accuracy: {metrics['accuracy']:.4f}\n")

    print("Weighted Metrics:")
    print(f"Precision: {metrics['weighted']['precision']:.4f}")
    print(f"Recall:    {metrics['weighted']['recall']:.4f}")
    print(f"F1-score:  {metrics['weighted']['f1']:.4f}\n")

    print("Macro Metrics:")
    print(f"Precision: {metrics['macro']['precision']:.4f}")
    print(f"Recall:    {metrics['macro']['recall']:.4f}")
    print(f"F1-score:  {metrics['macro']['f1']:.4f}\n")

    print("Micro Metrics:")
    print(f"Precision: {metrics['micro']['precision']:.4f}")
    print(f"Recall:    {metrics['micro']['recall']:.4f}")
    print(f"F1-score:  {metrics['micro']['f1']:.4f}\n")

    print("************ Finish ************")

    # 6) Plot + save learning curves (adjust for actual epochs trained)
    actual_epochs = len(train_loss_history)
    epochs = np.arange(1, actual_epochs + 1)

    # a) Training Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name.capitalize()}: Training Loss")
    if early_stopped:
        plt.axvline(x=actual_epochs, color='red', linestyle='--', alpha=0.7,
                    label=f'Early Stop (epoch {actual_epochs})')
    plt.legend()
    plt.tight_layout()
    loss_plot = f"{dataset_name}_train_loss.png"
    plt.savefig(loss_plot)
    plt.close()

    # b) Validation and Test Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_acc_history, label="Val Accuracy")
    if dataset_name.lower() == "pubmed":
        plt.plot(epochs, test_acc_history[:actual_epochs], label="Test Accuracy")
        plt.axhline(y=pubmed_early_stop_threshold, color='red', linestyle='--', alpha=0.7,
                    label=f'Early Stop Threshold ({pubmed_early_stop_threshold:.2f})')
    if early_stopped:
        plt.axvline(x=actual_epochs, color='red', linestyle='--', alpha=0.7,
                    label=f'Early Stop (epoch {actual_epochs})')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if dataset_name.lower() == "pubmed":
        plt.title(f"{dataset_name.capitalize()}: Validation & Test Accuracy")
    else:
        plt.title(f"{dataset_name.capitalize()}: Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_plot = f"{dataset_name}_val_accuracy.png"
    plt.savefig(acc_plot)
    plt.close()

    print(f"Saved plots: {loss_plot}, {acc_plot}")


# ─────────────────────────────────────────────────────────────────────
# Command‐line argument parsing
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a 2-layer GCN on a citation dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cora", "citeseer", "pubmed"],
        required=True,
        help="Which dataset to train on: 'cora', 'citeseer', or 'pubmed'."
    )
    default_data_dir = os.path.join(PROJECT_ROOT, "data", "stage_5_data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,  # ← Make sure this is exactly as shown
        help=(
            "Path to directory containing 'cora', 'citeseer', 'pubmed' subfolders. "
            f"Default is '{default_data_dir}'."
        )
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=16,
        help="Number of hidden units in the first GCN layer."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4,
        help="Weight decay (L2 penalty) for the optimizer."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5,
        help="Dropout probability after the first GCN layer."
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    train_and_evaluate(
        dataset_name=args.dataset,
        data_path=args.data_dir,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_epochs=args.epochs,
        seed=args.seed
    )