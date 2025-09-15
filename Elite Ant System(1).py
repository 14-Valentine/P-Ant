# --- Elitist Ant System (EAS) — clean & faithful to paper ---
# Keeps the original-style structure: parameters at top, helpers, then main loop.
# Removes path visualization; adds multi-elite_e comparison with best-so-far curves.

import random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS (edit here)
# -------------------------
random.seed(42)
np.random.seed(42)

n = 20                 # number of cities
m = n                  # ants per iteration (paper recommends m ~ n)
iters = 100            # algorithm iterations
alpha = 0.5            # pheromone weight    (rule (1))
beta  = 0.5            # visibility weight   (rule (1))
rho   = 0.5            # evaporation rate p  (rule (2))
Q     = 1.0            # pheromone constant  (rules (2),(3))
tau0  = 1.0            # initial pheromone τ0

# IMPORTANT: 'elite_e' is NOT real ants; it's a weight that reinforces T+ (eq. (3) in paper).
# You can match the figure in the paper: [0, 3, 5, 10, 30]
elite_values = [0, 1, 2, 3, 4, 5]

# -------------------------
# Build a random TSP instance (positions, distances, τ, η)
# -------------------------
def build_graph(n, tau0):
    pos = np.random.rand(n, 2)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.linalg.norm(pos[i] - pos[j])
    tau = np.full((n, n), tau0); np.fill_diagonal(tau, 0.0)
    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if D[i, j] > 0:
                eta[i, j] = 1.0 / D[i, j]
    return pos, D, tau, eta

# -------------------------
# Helpers
# -------------------------
def tour_length(order, D):
    L = 0.0; n = len(order)
    for k in range(n):
        i = order[k]; j = order[(k+1) % n]
        L += D[i, j]
    return L

def transition_probs(current, allowed, tau_snap, eta, alpha, beta):
    # Rule (1) in the paper: P_ij ∝ (τ_ij^α) * (η_ij^β)
    w = [(tau_snap[current, j] ** alpha) * (eta[current, j] ** beta) for j in allowed]
    s = sum(w)
    if s == 0.0:
        return [1.0 / len(allowed)] * len(allowed)
    return [x / s for x in w]

# -------------------------
# Core EAS runner: returns best-so-far curve per iteration
# -------------------------
def run_eas_best_curve(D, eta, iters, m, alpha, beta, rho, Q, tau0, elite_e):
    n = D.shape[0]
    tau = np.full((n, n), tau0); np.fill_diagonal(tau, 0.0)

    best_so_far = float('inf')
    best_curve = []
    T_plus = None  # best tour so far (T+ in paper)

    for _ in range(iters):
        tau_snap = tau.copy()
        tours, lengths = [], []

        # Construct solutions (ants build complete tours)
        for _a in range(m):
            start = np.random.randint(0, n)
            current = start
            visited = [start]
            while len(visited) < n:
                allowed = [j for j in range(n) if j not in visited]
                probs = transition_probs(current, allowed, tau_snap, eta, alpha, beta)  # (1)
                next_city = random.choices(allowed, weights=probs, k=1)[0]
                visited.append(next_city)
                current = next_city

            L = tour_length(visited, D)
            tours.append(visited); lengths.append(L)

            if L < best_so_far:
                best_so_far = L
                T_plus = visited.copy()

        # Evaporation + deposit from all ants (basic AS update) — eq. (2)
        tau *= (1.0 - rho); np.fill_diagonal(tau, 0.0)
        for tour, L in zip(tours, lengths):
            dep = Q / L
            for k in range(n):
                i = tour[k]; j = tour[(k+1) % n]
                tau[i, j] += dep; tau[j, i] += dep

        # Elitist reinforcement on the global best-so-far tour — eq. (3)
        if elite_e > 0 and T_plus is not None:
            dep_e = elite_e * Q / best_so_far
            for k in range(n):
                i = T_plus[k]; j = T_plus[(k+1) % n]
                tau[i, j] += dep_e; tau[j, i] += dep_e

        best_curve.append(best_so_far)

    return best_curve, best_so_far

# -------------------------
# EXPERIMENT: compare different elite_e on the same instance
# -------------------------
pos, D, tau_init, eta = build_graph(n, tau0)

curves = {}
finals = {}
baseline = float('inf')

for e in elite_values:
    print(f"Running EAS with elite_e = {e}")
    curve, bestL = run_eas_best_curve(D, eta, iters, m, alpha, beta, rho, Q, tau0, elite_e=e)
    curves[e] = curve; finals[e] = bestL
    baseline = min(baseline, bestL)
    print(f"  best-so-far length = {bestL:.4f}")

# -------------------------
# PLOT: like the paper — Length vs Iteration with multiple elite_e
# -------------------------
plt.figure(figsize=(8,5))

xs = np.arange(1, iters+1)
for e in elite_values:
    label = f"e={e}"
    plt.plot(xs, curves[e], label=label)

plt.hlines(baseline, xmin=1, xmax=iters, linestyles="dashed", colors="gray", label="best found")
plt.xlabel("Iteration")
plt.ylabel("Best tour length")
plt.title("EAS with different elite values")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
