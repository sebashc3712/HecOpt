"""
Example 1: Combinatorial DFL — Shortest Path on a Grid
=======================================================

Demonstrates a full end-to-end Predict-then-Optimize pipeline for the
shortest path problem using SPO+ and PFYL surrogate losses.

Pipeline:
    features ──► neural net ──► predicted edge costs
                                        │
                                  LP solver (SciPy)
                                        │
                              shortest-path solution
                                        │
                          SPO+ / PFYL surrogate loss
                                        │
                              backprop to neural net

Run:
    python examples/shortest_path_example.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hecopt import CombinatorialPtOLayer
from hecopt.baselines.shortest_path import ShortestPathDataset, ShortestPathModel

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GRID_SIZE = 5
N_FEATURES = 8
N_SAMPLES_TRAIN = 800
N_SAMPLES_TEST = 200
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 5e-3
METHOD = "spo_plus"        # "spo_plus" | "pfyl"
LAMBDA_HYBRID = 0.1        # MSE weight in hybrid loss
SEED = 42

torch.manual_seed(SEED)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
print("Generating data …")
train_ds = ShortestPathDataset(
    n_samples=N_SAMPLES_TRAIN,
    grid_size=GRID_SIZE,
    n_features=N_FEATURES,
    degree=2,               # quadratic feature-to-cost mapping (harder)
    noise_std=0.1,
    seed=SEED,
)
test_ds = ShortestPathDataset(
    n_samples=N_SAMPLES_TEST,
    grid_size=GRID_SIZE,
    n_features=N_FEATURES,
    degree=2,
    noise_std=0.1,
    seed=SEED + 1,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

n_edges = train_ds.model.n_vars

# --------------------------------------------------------------------------- #
# Model and DFL layer
# --------------------------------------------------------------------------- #
net = nn.Sequential(
    nn.Linear(N_FEATURES, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_edges),
)

opt_model = ShortestPathModel(GRID_SIZE)
layer = CombinatorialPtOLayer(
    opt_model,
    method=METHOD,
    lambda_hybrid=LAMBDA_HYBRID,
    n_samples=10,
    sigma=0.5,
)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
print(f"\nTraining with {METHOD.upper()} loss on {GRID_SIZE}×{GRID_SIZE} grid …\n")
print(f"{'Epoch':>5}  {'Train Loss':>12}  {'Test Regret':>12}")
print("-" * 35)

for epoch in range(1, N_EPOCHS + 1):
    net.train()
    total_loss = 0.0
    n_batches = 0

    for feats, costs, _ in train_loader:
        optimizer.zero_grad()
        theta_pred = net(feats)
        loss = layer.loss(theta_pred, costs)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    # Evaluation
    net.eval()
    regrets = []
    with torch.no_grad():
        for feats, costs, _ in test_loader:
            theta_pred = net(feats)
            regret = layer.regret(theta_pred, costs)
            regrets.append(regret.item())

    avg_loss = total_loss / n_batches
    avg_regret = sum(regrets) / len(regrets)
    print(f"{epoch:>5}  {avg_loss:>12.4f}  {avg_regret:>12.4f}")

# --------------------------------------------------------------------------- #
# Baseline: predict-then-optimize with MSE loss
# --------------------------------------------------------------------------- #
print("\n--- MSE Baseline (Predict-Then-Optimize without DFL) ---")
net_mse = nn.Sequential(
    nn.Linear(N_FEATURES, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_edges),
)
optimizer_mse = torch.optim.Adam(net_mse.parameters(), lr=LR)
mse_loss_fn = nn.MSELoss()
layer_eval = CombinatorialPtOLayer(opt_model, method="spo_plus")

for epoch in range(1, N_EPOCHS + 1):
    net_mse.train()
    for feats, costs, _ in train_loader:
        optimizer_mse.zero_grad()
        theta_pred = net_mse(feats)
        loss = mse_loss_fn(theta_pred, costs)
        loss.backward()
        optimizer_mse.step()

net_mse.eval()
regrets_mse = []
with torch.no_grad():
    for feats, costs, _ in test_loader:
        theta_pred = net_mse(feats)
        regret = layer_eval.regret(theta_pred, costs)
        regrets_mse.append(regret.item())

print(f"MSE baseline test regret: {sum(regrets_mse)/len(regrets_mse):.4f}")
print(f"\nDone.  DFL ({METHOD}) typically achieves lower regret than MSE baseline.")
