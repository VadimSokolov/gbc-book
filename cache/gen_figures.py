"""
Generate and save cached figures for the GBC book.
Run once from the gbc-book directory:
    python cache/gen_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import norm as scipy_norm

OUT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────
# fig-qtd-twostate  (ch06-distributional-rl)
# ─────────────────────────────────────────────────────────────────────
print("Generating fig-qtd-twostate ...")

np.random.seed(42)

gamma = 0.9
n_quantiles = 21
taus = np.linspace(0.5 / n_quantiles, 1 - 0.5 / n_quantiles, n_quantiles)
n_episodes = 10000
episode_length = 200
lr = 0.05

theta = np.zeros((2, n_quantiles))
history_steps, history_q10, history_q50, history_q90 = [], [], [], []

for ep in range(n_episodes):
    state = 0
    for t in range(episode_length):
        reward = 2.0 if (state == 0 and np.random.rand() < 0.5) else 0.0
        next_state = 1 - state
        for i in range(n_quantiles):
            j = np.random.randint(n_quantiles)
            target = reward + gamma * theta[next_state, j]
            delta = target - theta[state, i]
            grad = taus[i] if delta >= 0 else taus[i] - 1.0
            theta[state, i] += lr * grad
        state = next_state
    if ep % 50 == 0:
        history_steps.append(ep)
        history_q10.append(theta[0, 2])
        history_q50.append(theta[0, n_quantiles // 2])
        history_q90.append(theta[0, -3])

n_mc = 50000
mc_returns = np.zeros(n_mc)
for i in range(n_mc):
    ret, disc, state = 0.0, 1.0, 0
    for t in range(episode_length):
        r = 2.0 if (state == 0 and np.random.rand() < 0.5) else 0.0
        ret += disc * r
        disc *= gamma
        state = 1 - state
    mc_returns[i] = ret
mc_returns.sort()
mc_quantiles = np.array([mc_returns[int(t * n_mc)] for t in taus])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(taus, theta[0], 'o-', color='steelblue', markersize=4, label='Quantile TD (learned)')
axes[0].plot(taus, mc_quantiles, 's--', color='coral', markersize=4, label='Monte Carlo (ground truth)')
axes[0].set_xlabel(r'$\tau$')
axes[0].set_ylabel(r'$F_{Z(A)}^{-1}(\tau)$')
axes[0].set_title('Learned quantile function (state A)')
axes[0].legend(fontsize=8)
axes[1].plot(history_steps, history_q10, label=r'$\tau \approx 0.10$', color='#d62728')
axes[1].plot(history_steps, history_q50, label=r'$\tau \approx 0.50$', color='#2ca02c')
axes[1].plot(history_steps, history_q90, label=r'$\tau \approx 0.90$', color='#1f77b4')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Quantile estimate')
axes[1].set_title('Convergence of quantile estimates (state A)')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig-qtd-twostate.png", dpi=150, bbox_inches="tight")
plt.close()

output_qtd = (
    f"Learned mean return (state A): {theta[0].mean():.3f}\n"
    f"MC mean return (state A): {mc_returns.mean():.3f}\n"
    f"Learned std (state A): {np.std(np.diff(theta[0])):.4f}\n"
)
(OUT / "fig-qtd-twostate-output.txt").write_text(output_qtd)
print("  saved fig-qtd-twostate.png")
print(output_qtd)

# ─────────────────────────────────────────────────────────────────────
# fig-friedman-iqn  (ch05-iqn)
# ─────────────────────────────────────────────────────────────────────
print("Generating fig-friedman-iqn ...")

class IQN(nn.Module):
    def __init__(self, xdim, hdim=256, M=64):
        super().__init__()
        self.M = M
        self.fc_tau = nn.Sequential(nn.Linear(M, hdim), nn.ReLU())
        self.fc_x   = nn.Sequential(nn.Linear(xdim, hdim), nn.ReLU())
        self.head   = nn.Sequential(nn.Linear(hdim, hdim), nn.ReLU(),
                                    nn.Linear(hdim, 64),   nn.Tanh(),
                                    nn.Linear(64, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, tau):
        i = torch.arange(self.M, dtype=torch.float32)
        h_tau = self.fc_tau(torch.cos(i * torch.pi * tau))
        h_x   = self.fc_x(x)
        return self.head(h_x * h_tau.unsqueeze(0))


def pinball_loss(y, q_hat, tau):
    e = y - q_hat
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))


def composite_loss(model, x, y, weights=(0.70, 0.20, 0.10)):
    w1, w2, w3 = weights
    tau = torch.rand(1).item()
    q_hat = model(x, tau).squeeze(-1)
    L_pinball = pinball_loss(y, q_hat, tau)
    L_crps = sum(pinball_loss(y, model(x, t.item()).squeeze(-1), t.item())
                 for t in torch.rand(16)) / 16
    q_median = model(x, 0.5).squeeze(-1)
    L_mae = torch.mean(torch.abs(y - q_median))
    return w1 * L_pinball + w2 * L_crps + w3 * L_mae


def train_iqn(X_train, y_train, epochs=3000, hdim=256, M=64,
              lr=1e-3, wd=1e-4, weights=(0.70, 0.20, 0.10), seed=42):
    torch.manual_seed(seed)
    x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-8
    y_mean, y_std = float(y_train.mean()), float(y_train.std()) + 1e-8
    Xt = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    yt = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32)
    model = IQN(X_train.shape[1], hdim=hdim, M=M)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        composite_loss(model, Xt, yt, weights).backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    return model, x_mean, x_std, y_mean, y_std


def predict_quantiles(model, X_test, x_mean, x_std, y_mean, y_std, n_tau=500):
    tau_grid = np.linspace(0.005, 0.995, n_tau)
    Xt = torch.tensor((X_test - x_mean) / x_std, dtype=torch.float32)
    quantiles = []
    with torch.no_grad():
        for tau in tau_grid:
            q = model(Xt, float(tau)).squeeze(-1).numpy()
            quantiles.append(q * y_std + y_mean)
    return np.array(quantiles), tau_grid


def get_q(quantiles, tau_grid, level):
    return quantiles[np.argmin(np.abs(tau_grid - level)), :]


np.random.seed(123)
n_fried = 200
x_fried = np.random.uniform(0, 1, n_fried)
sigma_fried = 1.0 + 3.0 * x_fried
y_fried = 10 * np.sin(np.pi * x_fried) + sigma_fried * np.random.randn(n_fried)
X_fried = x_fried.reshape(-1, 1)

model_fried, xm_f, xs_f, ym_f, ys_f = train_iqn(
    X_fried, y_fried, epochs=1000, hdim=128, M=32, seed=7
)
print("  Friedman model trained.")

X_grid_f = np.linspace(0.01, 0.99, 300).reshape(-1, 1)
q_fried_preds, tau_f = predict_quantiles(model_fried, X_grid_f, xm_f, xs_f, ym_f, ys_f, n_tau=200)

q05_f = get_q(q_fried_preds, tau_f, 0.05)
q25_f = get_q(q_fried_preds, tau_f, 0.25)
q50_f = get_q(q_fried_preds, tau_f, 0.50)
q75_f = get_q(q_fried_preds, tau_f, 0.75)
q95_f = get_q(q_fried_preds, tau_f, 0.95)

xg = X_grid_f.ravel()
true_mean = 10 * np.sin(np.pi * xg)
true_sigma = 1.0 + 3.0 * xg
true_q05 = true_mean + scipy_norm.ppf(0.05) * true_sigma
true_q95 = true_mean + scipy_norm.ppf(0.95) * true_sigma

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.scatter(x_fried, y_fried, s=6, alpha=0.3, color='gray', zorder=1)
ax.fill_between(xg, q05_f, q95_f, alpha=0.15, color='steelblue', label='90% PI (IQN)')
ax.fill_between(xg, q25_f, q75_f, alpha=0.30, color='steelblue', label='50% PI (IQN)')
ax.plot(xg, q50_f, color='steelblue', linewidth=2, label='Median (IQN)')
ax.plot(xg, true_mean, 'k--', linewidth=1.5, label='True mean')
ax.plot(xg, true_q05, 'r:', linewidth=1, label='True 5%/95%')
ax.plot(xg, true_q95, 'r:', linewidth=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("IQN quantile fan — Friedman 1D with heteroscedastic noise")
ax.legend(fontsize=7, loc='upper left')
plt.tight_layout()
plt.savefig(OUT / "fig-friedman-iqn.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved fig-friedman-iqn.png")

print("Done. All figures saved to cache/")
