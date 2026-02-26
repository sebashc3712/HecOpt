"""
Example 2: Non-Linear DFL — Multi-Product Pricing Optimisation
==============================================================

Demonstrates end-to-end Decision-Focused Learning for a non-linear pricing
problem using KKT implicit differentiation.

Pipeline:
    market features ──► neural net ──► predicted demand params (a, ε)
                                                    │
                                    NLP solver (SciPy SLSQP)
                                                    │
                                     optimal prices p*(θ̂)
                                                    │
                            KKT adjoint backward pass
                                                    │
                                    gradient to neural net

The key insight: even though the NLP is non-convex, the KKT adjoint method
provides a valid gradient signal that encourages the network to predict
demand parameters that lead to better pricing decisions — not just parameters
that minimise MSE in parameter space.

Run:
    python examples/pricing_example.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hecopt import NonLinearPtOLayer
from hecopt.baselines.pricing import PricingDataset, PricingModel

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
N_PRODUCTS = 3
N_FEATURES = 8
N_SAMPLES_TRAIN = 600
N_SAMPLES_TEST = 200
BATCH_SIZE = 16
N_EPOCHS = 20
LR = 1e-3
TIKHONOV = 1e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
print("Generating pricing dataset …")
train_ds = PricingDataset(
    n_samples=N_SAMPLES_TRAIN,
    n_products=N_PRODUCTS,
    n_features=N_FEATURES,
    noise_std=0.05,
    seed=SEED,
)
test_ds = PricingDataset(
    n_samples=N_SAMPLES_TEST,
    n_products=N_PRODUCTS,
    n_features=N_FEATURES,
    noise_std=0.05,
    seed=SEED + 1,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

opt_model = train_ds.opt_model
n_params = opt_model.n_params  # 2 * N_PRODUCTS

# --------------------------------------------------------------------------- #
# Neural network (feature → demand parameters)
# --------------------------------------------------------------------------- #
net = nn.Sequential(
    nn.Linear(N_FEATURES, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_params),
    nn.Softplus(),           # Ensures outputs are positive (valid a, ε)
)

# --------------------------------------------------------------------------- #
# DFL layer (KKT backward)
# --------------------------------------------------------------------------- #
dfl_layer = NonLinearPtOLayer(opt_model, tikhonov=TIKHONOV)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# --------------------------------------------------------------------------- #
# Training loop — Decision-Focused Learning
# --------------------------------------------------------------------------- #
print(f"\nDFL Training (KKT implicit differentiation) — {N_PRODUCTS} products\n")
print(f"{'Epoch':>5}  {'Train Loss':>12}  {'Test Regret':>12}")
print("-" * 35)

for epoch in range(1, N_EPOCHS + 1):
    net.train()
    total_loss = 0.0
    n_batches = 0

    for feats, theta_true, _, _, obj_true in train_loader:
        optimizer.zero_grad()
        theta_pred = net(feats)
        loss = dfl_layer.decision_loss(theta_pred, theta_true)
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
        for feats, theta_true, _, _, _ in test_loader:
            theta_pred = net(feats)
            regret = dfl_layer.regret(theta_pred, theta_true)
            regrets.append(regret.item())

    avg_loss = total_loss / max(n_batches, 1)
    avg_regret = sum(regrets) / len(regrets)
    print(f"{epoch:>5}  {avg_loss:>12.4f}  {avg_regret:>12.4f}")

# --------------------------------------------------------------------------- #
# Baseline: MSE pre-training (Predict-Then-Optimize without DFL)
# --------------------------------------------------------------------------- #
print("\n--- MSE Baseline (standard supervised learning on demand params) ---")
net_mse = nn.Sequential(
    nn.Linear(N_FEATURES, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, n_params),
    nn.Softplus(),
)
optimizer_mse = torch.optim.Adam(net_mse.parameters(), lr=LR)
mse_fn = nn.MSELoss()

for epoch in range(1, N_EPOCHS + 1):
    net_mse.train()
    for feats, theta_true, _, _, _ in train_loader:
        optimizer_mse.zero_grad()
        theta_pred = net_mse(feats)
        loss = mse_fn(theta_pred, theta_true)
        loss.backward()
        optimizer_mse.step()

net_mse.eval()
regrets_mse = []
with torch.no_grad():
    for feats, theta_true, _, _, _ in test_loader:
        theta_pred = net_mse(feats)
        regret = dfl_layer.regret(theta_pred, theta_true)
        regrets_mse.append(regret.item())

print(f"MSE baseline test regret: {sum(regrets_mse)/len(regrets_mse):.4f}")
print("\nDone. DFL should achieve lower decision regret than the MSE baseline.")
