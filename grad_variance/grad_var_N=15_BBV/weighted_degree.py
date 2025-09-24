import pennylane as qml
import torch
import numpy as np
import os, pandas as pd, random, gc
from collections import defaultdict
from typing import List, Tuple, Dict

# === 基本設定 ===
N = 15
m = 2
file_name = "maQAOA_pi8init"
p = 1
PI = torch.pi

device = torch.device("cpu")

NUM_TRIAL = 5000






#--------------------------------------------------#
def generate_case(seed=42):
    def generate_bbv_graph(
        N: int,
        m: int,
        seed: int | None = None,
        m0: int | None = None,
        w0: float = 1.0,
        delta: float = 1.0
    ) -> Tuple[List[Tuple[int, int]], List[float], List[float], List[int]]:
        """
        產生 BBV (Barrat–Barthelemy–Vespignani) 加權網路。

        回傳：
        edges:   [(u,v), ...]（u<v）
        weights: [w_uv, ...] 與 edges 對齊
        strength: [s_i]  每個節點的加權度數 s_i = sum_j w_ij
        """
        assert 1 <= m < N, "需滿足 1 <= m < N"
        if m0 is None:
            m0 = m
        assert m0 >= m and m0 <= N
        assert w0 >= 0 and delta >= 0
        rng = random.Random(seed)

        adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        strength = [0.0] * N

        # 初始 K_{m0}
        for i in range(m0):
            for j in range(i + 1, m0):
                adj[i][j] = w0
                adj[j][i] = w0
                strength[i] += w0
                strength[j] += w0

        def weighted_sample_without_replacement(pop_indices: List[int], weights: List[float], k: int) -> List[int]:
            if sum(weights) <= 0:
                return rng.sample(pop_indices, k)
            chosen = []
            pool = pop_indices[:]
            w = weights[:]
            for _ in range(k):
                pick = rng.choices(pool, weights=w, k=1)[0]
                idx = pool.index(pick)
                chosen.append(pick)
                pool.pop(idx); w.pop(idx)
            return chosen

        for new_v in range(m0, N):
            candidates = list(range(new_v))
            weights_s = [strength[u] for u in candidates]
            targets = weighted_sample_without_replacement(candidates, weights_s, m)

            for u in targets:
                s_u_old = strength[u]
                if s_u_old > 0 and delta > 0:
                    increments = []
                    for v, w_uv in adj[u].items():
                        inc = delta * w_uv / s_u_old
                        if inc > 0:
                            increments.append((v, inc))
                    for v, inc in increments:
                        adj[u][v] += inc
                        adj[v][u] += inc
                        strength[u] += inc
                        strength[v] += inc

                if u not in adj[new_v]:
                    adj[new_v][u] = 0.0
                    adj[u][new_v] = 0.0
                adj[new_v][u] += w0
                adj[u][new_v] += w0
                strength[new_v] += w0
                strength[u]     += w0

        edges = []
        weights = []
        for u in range(N):
            for v, w in adj[u].items():
                if u < v:
                    edges.append((u, v))
                    weights.append(w)
        return edges, weights, strength

    edges, weights, deg_w = generate_bbv_graph(
        N=N, m=m, seed=seed, m0=None, w0=1.0, delta=1
    )
    weights = torch.tensor(weights, device=device, dtype=torch.float32)

    # --- 成本向量（為了計算 MaxCut 最佳值，用暴力法） ---
    num_states = 2 ** N
    states = torch.arange(num_states, device=device).unsqueeze(1)
    bits = ((states >> torch.arange(N, device=device)) & 1).bool()  # LSB->MSB
    costs = torch.zeros(num_states, device=device, dtype=torch.float32)
    for idx, (i, j) in enumerate(edges):
        diff = bits[:, i] ^ bits[:, j]
        costs += weights[idx] * diff.float()

    print(f"Min: {costs.min():.4f}, Mean: {costs.mean():.4f}, Max: {costs.max():.4f}")

    # --- 也建立 expval 用的 Hamiltonian（常數 + ZZ 形式） ---
    coeffs = []
    observables = []
    for (i, j), w in zip(edges, weights.tolist()):
        coeffs.append(0.5 * w)                        # 常數項 * I
        observables.append(qml.Identity(0))
        coeffs.append(-0.5 * w)                       # -0.5 w Z_i Z_j
        observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
    cost_hamiltonian = qml.Hamiltonian(coeffs, observables)

    return weights, edges, deg_w, costs, cost_hamiltonian

#--------------------------------------------------#

for i in range(50):
    weights, edges, deg_w, costs, cost_hamiltonian = generate_case(i)
    os.makedirs(f'weighted_degree', exist_ok=True)
    pd.DataFrame(np.array(deg_w)).to_csv(f'weighted_degree/{i}.csv', index=False, header=False)