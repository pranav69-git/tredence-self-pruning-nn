"""
Self-Pruning Neural Network on CIFAR-10
========================================
Implements learnable gate parameters per weight.
Gates trained with L1 sparsity penalty to push toward 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ──────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gate parameters.

    Each weight w_ij has a corresponding gate score g_ij.
    Gate = sigmoid(g_ij), so gate ∈ (0, 1).
    Effective weight = w_ij * gate_ij.

    L1 penalty on gates encourages them toward 0,
    effectively pruning the associated weight connection.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight + bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as weight, also learnable
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_params()

    def _init_params(self):
        # Kaiming init for weights (good for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        # Init gate scores near 0 → sigmoid ≈ 0.5 (all gates ~active at start)
        nn.init.zeros_(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gates ∈ (0,1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise gate the weights
        pruned_weights = self.weight * gates

        # Standard affine transform — gradients flow through both weight + gate_scores
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of all gate values for this layer."""
        return torch.sigmoid(self.gate_scores).sum()


# ──────────────────────────────────────────
# Part 2: Network using PrunableLinear
# ──────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward net for CIFAR-10 (32×32×3 = 3072 input features).
    All linear layers are PrunableLinear.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,  128)
        self.fc4 = PrunableLinear(128,  10)   # 10 CIFAR-10 classes
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4]

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum L1 of gates across all prunable layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def all_gate_values(self) -> torch.Tensor:
        """Concatenate all gate values for analysis."""
        return torch.cat([layer.get_gates().view(-1) for layer in self.prunable_layers()])

    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (effectively pruned)."""
        gates = self.all_gate_values()
        pruned = (gates < threshold).sum().item()
        return pruned / gates.numel() * 100.0


# ──────────────────────────────────────────
# Part 3: Data Loading
# ──────────────────────────────────────────

def get_dataloaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────
# Part 3: Training Loop
# ──────────────────────────────────────────

def train_epoch(model, loader, optimizer, lambda_sparse, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Sparsity regularization: penalize active gates
        sparse_loss = model.total_sparsity_loss()

        # Total loss
        loss = cls_loss + lambda_sparse * sparse_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total * 100.0


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100.0


def run_experiment(lambda_sparse: float, epochs: int, device, train_loader, test_loader, verbose: bool = True):
    """Train one model with given lambda. Return (test_acc, sparsity_pct, gate_values)."""
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  λ = {lambda_sparse}")
        print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, lambda_sparse, device)
        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            sparsity = model.compute_sparsity()
            print(f"  Epoch {epoch:3d} | Loss {train_loss:.4f} | "
                  f"Train Acc {train_acc:.1f}% | Sparsity {sparsity:.1f}%")

    test_acc  = evaluate(model, test_loader, device)
    sparsity  = model.compute_sparsity()
    gate_vals = model.all_gate_values().numpy()

    if verbose:
        print(f"\n  → Test Accuracy : {test_acc:.2f}%")
        print(f"  → Sparsity Level: {sparsity:.2f}%")

    return test_acc, sparsity, gate_vals


# ──────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────

def plot_gate_distribution(gate_vals: np.ndarray, lambda_val: float, save_path: str):
    """Histogram of gate values. Good pruning → spike at 0, cluster away from 0."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gate_vals, bins=100, color='steelblue', edgecolor='none', alpha=0.85)
    ax.set_xlabel('Gate Value', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(f'Gate Value Distribution  (λ = {lambda_val})', fontsize=14)
    ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1.5, label='Prune threshold (0.01)')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_all_distributions(results: dict, save_path: str):
    """Side-by-side gate distributions for all λ values."""
    lambdas = list(results.keys())
    n = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=False)

    for ax, lam in zip(axes, lambdas):
        gate_vals = results[lam]['gates']
        acc  = results[lam]['acc']
        spar = results[lam]['sparsity']
        ax.hist(gate_vals, bins=100, color='steelblue', edgecolor='none', alpha=0.85)
        ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1.5, label='Threshold')
        ax.set_xlabel('Gate Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'λ={lam}\nAcc={acc:.1f}%  Sparsity={spar:.1f}%', fontsize=12)
        ax.legend(fontsize=10)

    plt.suptitle('Gate Distributions Across λ Values', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined plot saved → {save_path}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    EPOCHS = 30           # Balance speed vs quality
    BATCH  = 128
    LAMBDAS = [1e-5, 1e-3, 1e-1]   # Low / Medium / High sparsity

    train_loader, test_loader = get_dataloaders(BATCH)

    results = {}
    for lam in LAMBDAS:
        test_acc, sparsity, gate_vals = run_experiment(
            lambda_sparse=lam,
            epochs=EPOCHS,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            verbose=True,
        )
        results[lam] = {'acc': test_acc, 'sparsity': sparsity, 'gates': gate_vals}

    # ── Print summary table ──
    print("\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("-"*42)
    for lam in LAMBDAS:
        r = results[lam]
        print(f"{lam:<12} {r['acc']:>14.2f} {r['sparsity']:>14.2f}")
    print("="*55)

    # ── Plots ──
    best_lam = max(results, key=lambda l: results[l]['acc'])
    plot_gate_distribution(
        gate_vals=results[best_lam]['gates'],
        lambda_val=best_lam,
        save_path='gate_distribution_best.png',
    )
    plot_all_distributions(results, save_path='gate_distributions_all.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
