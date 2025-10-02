import pennylane as qml
import torch
import numpy as np
import os, pandas as pd, random, gc
from collections import defaultdict
from typing import List, Tuple, Dict

cuda_id = '1'

# === 基本設定 ===
N = 15
m = 2
file_name = "maQAOA_group_param_and_reset_momentum"
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
def init_params(p: int, N: int, M: int,
                            seed: int = 42) -> np.ndarray:
    rng = random.Random(seed)
    kspace = list(range(-8, 9))

    params = [rng.choice(kspace)*(np.pi/8.0) for _ in range(p*(M+N))]
    
    return np.array(params, dtype=np.float32)

#--------------------------------------------------#
def build_masks_by_stages(edges, deg_order, p, N):

    # incident edge set builder
    def incident_set(S):
        idxs = []
        for e_idx, (u, v) in enumerate(edges):
            if u in S or v in S:
                idxs.append(e_idx)
        return set(idxs)

    M = len(edges)
    stages = []

    for stage in range(1, NUM_STAGES+1):
        cut = stage*(len(deg_order))//NUM_STAGES
        S = set(deg_order[:cut])
        E = incident_set(S)
        beta_mask = torch.zeros((p, N), device=device)
        for l in range(p):
            for q in S: beta_mask[l, q] = 1.0
        gamma_mask = torch.zeros((p, M), device=device)
        for l in range(p):
            for e in E: gamma_mask[l, e] = 1.0
        stages.append((beta_mask, gamma_mask))

    return stages

#--------------------------------------------------#
def run_case(weights, edges, costs, cost_hamiltonian, deg_w, 
             stages_masks, init_params, instance_id, trial_id):
    """
    - 參數排列: [gammas (p*M), betas (p*N)]
    - γ/β LLRD：依 stages_masks 的 (beta_mask, gamma_mask) 逐階段解凍
    """
    M = len(edges)
    betas_len = p * N
    total_len = p*M + betas_len

    deg_w = sorted(deg_w)

    dev = qml.device("lightning.gpu", wires=N, shots=None)
    params = torch.nn.Parameter(torch.tensor(init_params, device=device), requires_grad=True)

    @qml.qnode(dev, interface='torch')
    def qaoa_circuit(params):
        gammas = params[:p*M].view(p, M)
        betas  = params[p*M:].view(p, N)

        for i in range(N):
            qml.Hadamard(wires=i)

        for layer in range(p):
            for e_idx, (i, j) in enumerate(edges):
                qml.CNOT(wires=(i, j))
                # 注意：仍是 RZ(w_e * gamma) 的寫法；若想穩定，可把參數直接換成 theta_e = w_e * gamma_e
                qml.RZ(weights[e_idx] * gammas[layer, e_idx], wires=j)
                qml.CNOT(wires=(i, j))
            for i in range(N):
                qml.RX(betas[layer, i], wires=i)

        return qml.expval(cost_hamiltonian)

    def loss_fn(params):
        return -qaoa_circuit(params)

    best_para = params.detach().clone()
    max_exp = float('-inf')
    big_loss_list= []

    # ---- 掛 gradient hook：同時做「解凍遮罩」----
    gamma_mask_tensor = torch.zeros((p, M), device=device)
    beta_mask_tensor  = torch.zeros((p, N), device=device)

    def grad_hook(grad):
        # 拼成同長度遮罩
        gmask = gamma_mask_tensor.reshape(-1)
        bmask = beta_mask_tensor.reshape(-1)
        # 把未解凍項目乘 0
        scaled = torch.cat((gmask, bmask)) * grad
        return scaled

    params.register_hook(grad_hook)

    # ---- 三個 Stage：0/1/2 ----
    for stage_id, (beta_mask, gamma_mask) in enumerate(stages_masks):
        # 更新當前 stage 的遮罩（會影響 hook）
        beta_mask_tensor.copy_(beta_mask)
        gamma_mask_tensor.copy_(gamma_mask)

        # 重新啟用 requires_grad（清掉過往動量）
        params = torch.nn.Parameter(params.detach().clone(), requires_grad=True)
        params.register_hook(grad_hook)

        # Adam 訓練
        opt = torch.optim.Adam([params], lr=ADAM_LR*(sum(deg_w)/sum(deg_w[-(stage_id+1)*(N//NUM_STAGES):])))
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

        for epoch in range(1, MAX_EPOCH + 1):
            opt.zero_grad()
            loss = loss_fn(params)
            loss.backward()
            opt.step()

            lval = loss.item()
            big_loss_list.append(lval); loss_list.append(lval)

            if -lval > max_exp:
                max_exp = -lval
                best_para = params.detach().clone()

            #print(f"[instance:{instance_id}][trial:{trial_id}][stage:{stage_id}] "
            #      f"Epoch {epoch} | approx_ratio {-lval/costs.max():.6f}")

            last_stage = (stage_id == NUM_STAGES-1)
            if  ((last_stage)     and is_converged(10, TOL)) or \
                ((not last_stage) and is_converged_diff(10, PRETRAIN_SLOPE)) or\
                (len(big_loss_list)) == MAX_EPOCH:
                break

        #print(f"[instance:{instance_id}][trial:{trial_id}][stage:{stage_id}] "
        #      f"best approx_ratio:{max_exp/costs.max()}")

    print(f"[instance:{instance_id}][trial:{trial_id}] best approx_ratio:{max_exp/costs.max()}")
    return [best_para.tolist()], [max_exp/costs.max().tolist()], [len(big_loss_list)], [big_loss_list]

#--------------------------------------------------#

for i in range(20):

    os.makedirs(f'{file_name}/{i}', exist_ok=True)

    weights, edges, deg_w, costs, cost_hamiltonian = generate_case(i)
    M = len(edges)

    # 依（加權）度數由大到小的排序
    deg_order = list(np.argsort(-np.array(deg_w)))
    # 建立三個 stage 的 γ/β 解凍遮罩
    stages_masks = build_masks_by_stages(edges, deg_order, p, N)
    
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
            weights, edges, costs, cost_hamiltonian, deg_w, 
            stages_masks, params, i, trial_id
        )

        pd.DataFrame(best_params_list).to_csv(f'{file_name}/{i}/best_params.csv', mode='a', index=False, header=False)
        pd.DataFrame(best_approx_list).to_csv(f'{file_name}/{i}/best_approx.csv', mode='a', index=False, header=False)
        pd.DataFrame(num_epoch_list).to_csv(f'{file_name}/{i}/num_epoch.csv', mode='a', index=False, header=False)
        pd.DataFrame(loss_list_list).to_csv(f'{file_name}/{i}/loss_list.csv', mode='a', index=False, header=False)

    del weights, edges, deg_w, costs, cost_hamiltonian
    gc.collect()
    torch.cuda.empty_cache()
