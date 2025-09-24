import pennylane as qml
import torch
import numpy as np
import os, pandas as pd, random, gc
from collections import defaultdict
from typing import List, Tuple, Dict

cuda_id = '0'

# === 基本設定 ===
N = 12
m = 2
file_name = "maQAOA_reset_momentum"
p = 1
PI = torch.pi

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id
device = torch.device("cuda")

# === 初始化與優化設定 ===
ADAM_LR = 0.01
TOL = 1e-6
MAX_EPOCH = 10000
PRETRAIN_SLOPE = 0.5
NUM_STAGES = 3

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
def init_params(p: int, N: int, M: int,
                            seed: int = 42) -> np.ndarray:
    rng = random.Random(seed)
    kspace = list(range(-8, 9))

    params = [rng.choice(kspace)*(np.pi/8.0) for _ in range(p*(M+N))]
    
    return np.array(params, dtype=np.float32)

#--------------------------------------------------#
def run_case(weights, edges, costs, cost_hamiltonian, init_params, instance_id, trial_id):
    M = len(edges)
    dev = qml.device("lightning.gpu", wires=N, shots=None)
    params = torch.nn.Parameter(torch.tensor(init_params, device=device), requires_grad=True)

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

    best_para = params
    max_exp = float('-inf')
    big_loss_list= []

    for stage_id in range(NUM_STAGES):
        
        params = torch.nn.Parameter(params.detach().clone(), requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=ADAM_LR)
        loss_list = []

        def is_converged(window=10, tol=TOL):
            if len(loss_list) < window:
                return False
            recent = loss_list[-window:]
            return (max(recent) - min(recent)) < tol * weights.sum().detach().item()
        
        def is_converged_diff(window=10, ratio=PRETRAIN_SLOPE):
                if len(loss_list) < 2*window:
                    return False
                diff_past_mean = (loss_list[0] - loss_list[-1]) / (len(loss_list)-1)
                diff_now = (loss_list[-window] - loss_list[-1]) / (window-1)
                return diff_now < diff_past_mean * ratio

        for epoch in range(1, MAX_EPOCH+1):
            optimizer.zero_grad()
            loss = loss_fn(params)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if -loss_val > max_exp:
                max_exp = -loss_val
                best_para = params.detach().clone()

            loss_list.append(loss_val)
            big_loss_list.append(loss_val)
            #print(f"[instance:{instance_id}][trial:{trial_id}][stage:{stage_id}] Epoch {epoch} | approximate ratio {-loss_val/costs.max():.6f}")

            last_stage = (stage_id == NUM_STAGES-1)
            if  ((last_stage)     and is_converged(10, TOL)) or \
                ((not last_stage) and is_converged_diff(10, PRETRAIN_SLOPE)) or\
                (len(big_loss_list)) == MAX_EPOCH:
                break

    print(f"[instance:{instance_id}][trial:{trial_id}] best approximate ratio:{max_exp/costs.max()}")

    return [best_para.tolist()], [max_exp/costs.max().tolist()], [len(big_loss_list)], [big_loss_list]

#--------------------------------------------------#

for i in range(20):

    os.makedirs(f'{file_name}/{i}', exist_ok=True)

    weights, edges, deg_w, costs, cost_hamiltonian = generate_case(i)
    M = len(edges)

    for trial_id in range(100):
        
        if os.path.exists(f'{file_name}/{i}/best_params.csv'):
            data = pd.read_csv(f'{file_name}/{i}/best_params.csv', header=None)
            data = np.array(data[0])
            if len(data)!=trial_id:
                continue

        params = init_params(
            p=p, N=N, M=M,
            seed=trial_id
        )
        
        best_params_list, best_approx_list, num_epoch_list, loss_list_list = run_case(
            weights, edges, costs, cost_hamiltonian, params, i, trial_id
        )

        pd.DataFrame(best_params_list).to_csv(f'{file_name}/{i}/best_params.csv', mode='a', index=False, header=False)
        pd.DataFrame(best_approx_list).to_csv(f'{file_name}/{i}/best_approx.csv', mode='a', index=False, header=False)
        pd.DataFrame(num_epoch_list).to_csv(f'{file_name}/{i}/num_epoch.csv', mode='a', index=False, header=False)
        pd.DataFrame(loss_list_list).to_csv(f'{file_name}/{i}/loss_list.csv', mode='a', index=False, header=False)

    del weights, edges, deg_w, costs, cost_hamiltonian
    gc.collect()
    torch.cuda.empty_cache()
