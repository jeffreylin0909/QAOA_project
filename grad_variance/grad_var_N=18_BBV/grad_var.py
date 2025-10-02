import pennylane as qml
import torch
import numpy as np
import os, pandas as pd, random, gc
from collections import defaultdict
from typing import List, Tuple, Dict

cuda_id = input('Enter cuda id:')

# === 基本設定 ===
N = 18
m = 2
file_name = "maQAOA_pi8init"
p = 1
PI = torch.pi

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id
device = torch.device("cuda")

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
    ):
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

        for new_v in range(m0, N):
            chosen = set()
            for _ in range(m):
                # 每次都用「最新」的 strength 重新計算抽樣機率
                candidates = [u for u in range(new_v) if u not in chosen]
                weights_s = [strength[u] for u in candidates]

                # 抽 1 個
                u = rng.choices(candidates, weights=weights_s, k=1)[0]
                chosen.add(u)

                # ---- 立刻做「重分配」(只對 u 的舊鄰邊；不含 new_v) ----
                s_u_old = strength[u]
                if s_u_old > 0 and delta > 0:
                    increments = []
                    for v, w_uv in adj[u].items():
                        inc = delta * w_uv / s_u_old
                        if inc > 0 and v != new_v:
                            increments.append((v, inc))
                    for v, inc in increments:
                        adj[u][v] += inc
                        adj[v][u] += inc
                        strength[u] += inc
                        strength[v] += inc

                # ---- 立刻加新邊 (new_v, u) 並更新強度 ----
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
def get_grad(weights, edges, cost_hamiltonian, seed=42):
    M = len(edges)
    dev = qml.device("lightning.gpu", wires=N, shots=None)

    rng = random.Random(seed)
    params = torch.nn.Parameter(torch.tensor([rng.uniform(-PI, PI) for _ in range(p*(M+N))], device=device), requires_grad=True)

    # === ma-QAOA 電路 ===
    @qml.qnode(dev, interface='torch')
    def qaoa_circuit(params):
        # 參數切片：每層 [gammas(M), betas(N)]
        gammas = params[:p*M].view(p, M)
        betas  = params[p*M:].view(p, N)

        # 均勻疊加
        for i in range(N):
            qml.Hadamard(wires=i)

        for layer in range(p):
            # Cost: 每條邊有自己的 γ_{layer,e}
            for e_idx, (i, j) in enumerate(edges):
                # CNOT - RZ - CNOT 等效於 exp(-i * (theta/2) * Z_i Z_j)
                # 這裡沿用你的寫法：theta = w_e * gamma_{l,e}
                qml.CNOT(wires=(i, j))
                qml.RZ(weights[e_idx] * gammas[layer, e_idx], wires=j)
                qml.CNOT(wires=(i, j))
            # Mixer: 每個頂點有自己的 β_{layer,v}
            for i in range(N):
                qml.RX(betas[layer, i], wires=i)
        return qml.expval(cost_hamiltonian)

    def loss_fn(params):
        # 最大化期望值 => 最小化負值
        return -qaoa_circuit(params)
    
    loss = loss_fn(params)
    loss.backward()
    grads = params.grad.clone()   # grads 與 params 對齊（先 gamma 區塊、再 beta 區塊）

    return grads.cpu().tolist()
#--------------------------------------------------#

def get_grad_var(weights, edges, cost_hamiltonian):
    M = len(edges)
    grad_list = []
    for i in range(NUM_TRIAL):
        
        grad_list.append(get_grad(weights, edges, cost_hamiltonian, i)[p*M:])
    grad_list = np.array(grad_list)
    return np.var(grad_list, axis=0)

#--------------------------------------------------#

for i in range(50):
    weights, edges, deg_w, costs, cost_hamiltonian = generate_case(i)
    print(f"runnung instance {i}")
    grad_var = get_grad_var(weights, edges, cost_hamiltonian)

    os.makedirs(f'grad_var', exist_ok=True)
    pd.DataFrame(np.array(grad_var)).to_csv(f'grad_var/{i}.csv', index=False, header=False)