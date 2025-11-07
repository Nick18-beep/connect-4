# trainer.py

import os, importlib.util, inspect, sys
from collections import namedtuple
from multiprocessing import Process, Queue
from typing import Tuple, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4 import C4State
from model import PolicyValueNet

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def _choose_device() -> str:
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device detected: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("[INFO] No CUDA device detected.")
    return "cpu"

_AUTO_DEVICE = _choose_device()

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    allow_tf32 = (major >= 8)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")

Sample = namedtuple('Sample', 'obs pi z')
SamplePER = namedtuple('SamplePER', 'idx obs pi z weight')

def _safe_probs_from_pi(pi: np.ndarray, legal: list) -> np.ndarray:
    arr = np.array([pi[a] for a in legal], dtype=np.float32)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones(len(legal), dtype=np.float32) / max(1, len(legal))
    arr = arr / s
    return arr / float(arr.sum())

def self_play_worker(params_q: Queue, out_q: Queue, device: str, sims: int, g_scale: float, seed: int, use_gumbel_actions: bool, use_dynamic_k: bool):
    import os as _os, torch as _torch, numpy as _np, random as _random
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _torch.set_num_threads(1)

    from model import PolicyValueNet as _PVN
    from mcts_gumbel import GumbelMCTS as _GMCTS

    try:
        mod = sys.modules[_GMCTS.__module__].__file__
        print(f"[worker] MCTS class from: {mod}")
    except Exception:
        pass

    _torch.manual_seed(seed); _np.random.seed(seed); _random.seed(seed)
    model = _PVN().to(device)
    model.eval()

    params = inspect.signature(_GMCTS.__init__).parameters
    kw = dict(model=model, num_simulations=sims, device=device)
    opt = {
        'root_dirichlet_alpha': 0.3,
        'root_dirichlet_frac' : 0.01, #0.25
        'c_puct'             : 1.0,
        'c_init'             : 1.25,
        'c_base'             : 19652.0,
        'fpu_reduction'      : 0.2,
        'gumbel_scale'       : g_scale,
    }
    for k, v in opt.items():
        if k in params: kw[k] = v
    
    def maybe_sync():
        try:
            state_dict = params_q.get_nowait()
            model.load_state_dict(state_dict)
            model.eval()
        except Exception:
            pass

    while True:
        maybe_sync()
        state = C4State.initial()
        history = []
        mcts = _GMCTS(**kw)

        while True:
            ply = state.ply

            # dynamic Gumbel Top-k
            current_k = 1
            if use_dynamic_k:
                if ply < 12:
                    current_k = 3
                elif ply < 24:
                    current_k = 2

            temp = 1.0 if ply < 10 else 1e-6
            pi, _ = mcts.run(state, temperature=temp)
            legal = state.legal_actions()
            if not legal:
                break

            if use_gumbel_actions and len(legal) > 1:
                q_vals = mcts.get_root_q_values()
                gumbels = _np.random.gumbel(loc=0, scale=g_scale, size=len(legal))
                noisy_q = {a: q_vals.get(a, -1e9) + gumbels[i] for i, a in enumerate(legal)}
                sorted_actions = sorted(legal, key=lambda a: noisy_q[a], reverse=True)
                top_k_actions = sorted_actions[:min(current_k, len(sorted_actions))]

                if current_k == 1:
                    a = top_k_actions[0]
                else:
                    top_k_pi = {action: float(pi[action]) for action in top_k_actions}
                    pi_sum = sum(top_k_pi.values())
                    if pi_sum > 1e-9:
                        probs_array = _np.array(list(top_k_pi.values()), dtype=_np.float32)
                        probs_array = probs_array / _np.sum(probs_array)
                        a = int(_np.random.choice(list(top_k_pi.keys()), p=probs_array))
                    else:
                        a = int(_np.random.choice(top_k_actions))
            else:
                probs = _safe_probs_from_pi(pi, legal)
                a = int(_np.random.choice(legal, p=probs))

            history.append((state.to_planes(), pi, state.player))
            state = state.apply(a)
            mcts.update_root(a)

            term, outcome = state.terminal()
            if term:
                z = 0 if outcome == 0 else outcome
                for obs, pi_h, player in history:
                    try:
                        out_q.put_nowait(Sample(obs=obs, pi=pi_h, z=float(z * player)))
                    except Full:
                        # Drop the sample if the queue is congested to keep workers responsive
                        break
                break

def _triton_available() -> bool:
    spec = importlib.util.find_spec("triton")
    return spec is not None

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # MODIFIED: Changed dtype to np.float64 to prevent numerical drift
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64) 
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0

    def push(self, priority: float, data: Sample):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> Tuple[int, float, object]:
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[cl_idx]:
                parent_idx = cl_idx
            else:
                v -= self.tree[cl_idx]
                parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, float(self.tree[leaf_idx]), self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])

class ReplayBuffer:
    def __init__(self, capacity: int, per_alpha: float = 0.6, per_beta: float = 0.4):
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = (1.0 - per_beta) / 100_000
        self.epsilon = 1e-6
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(self, sample: Sample):
        self.tree.push(self.max_priority, sample)

    def sample(self, batch_size: int) -> SamplePER:
        if self.tree.size == 0:
            return SamplePER(np.empty(0, dtype=np.int64),
                             np.empty((0,4,6,7), np.float32),
                             np.empty((0,7), np.float32),
                             np.empty((0,), np.float32),
                             np.empty(0, dtype=np.float32))
        obs_b, pi_b, z_b = [], [], []
        idxs_b, weights_b = [], []
        segment = self.tree.total_priority / batch_size
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        base = segment * np.arange(batch_size, dtype=np.float64)
        offsets = np.random.uniform(0.0, segment, size=batch_size)
        sample_points = base + offsets

        for v in sample_points:
            idx, p, data = self.tree.get_leaf(v)
            sampling_prob = p / max(1e-8, self.tree.total_priority)
            
            # MODIFICATO: Aggiunto clipping con np.maximum per evitare la divisione per zero
            # quando sampling_prob è 0, garantendo stabilità numerica.
            base_for_power = self.tree.size * sampling_prob
            weight = np.power(np.maximum(base_for_power, self.epsilon), -self.per_beta)

            idxs_b.append(idx)
            weights_b.append(weight)
            obs_b.append(data.obs)
            pi_b.append(data.pi)
            z_b.append(data.z)

        max_w = max(weights_b) if weights_b else 1.0
        weights_norm = np.array(weights_b, dtype=np.float32) / max_w

        obs = np.stack(obs_b).astype('float32')
        pi  = np.stack(pi_b).astype('float32')
        z   = np.array(z_b, dtype='float32')
        return SamplePER(idxs_b, obs, pi, z, weights_norm)

    def update_priorities(self, tree_indices: list, td_errors: np.ndarray):
        priorities = np.power(np.abs(td_errors) + self.epsilon, self.per_alpha)
        self.max_priority = max(self.max_priority, float(np.max(priorities))) if priorities.size else self.max_priority
        for i, p in zip(tree_indices, priorities):
            self.tree.update(int(i), float(p))

    def __len__(self):
        return self.tree.size

class Trainer:
    def __init__(self, device="auto", lr=1e-6, weight_decay=1e-4,
                 sims=160, gumbel_scale=1.0, compile_model=False, compile_backend="inductor",
                 updates_per_iter=8, value_loss_weight: float = 1.0,
                 grad_clip: float = 2.0, scheduler_Tmax: int = 30_000,
                 ema_decay: float = 0.997, use_ema: bool = True,
                 policy_entropy_weight: float = 1e-3,
                 smooth_value_loss: bool = True,
                 per_alpha: float = 0.6, per_beta: float = 0.4):
        if device == "auto":
            device = _AUTO_DEVICE
        self.device = device
        self.model = PolicyValueNet().to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.buffer = ReplayBuffer(capacity=500_000, per_alpha=per_alpha, per_beta=per_beta)
        self.sims = sims
        self.gumbel_scale = gumbel_scale
        self.updates_per_iter = int(updates_per_iter)
        self.value_loss_weight = float(value_loss_weight)
        self.grad_clip = float(grad_clip)
        self.policy_entropy_weight = float(policy_entropy_weight)
        self.smooth_value_loss = bool(smooth_value_loss)
        self.ema_decay = float(ema_decay)
        self.use_ema = bool(use_ema and 0.0 < self.ema_decay < 1.0)
        self.ema_model: Optional[PolicyValueNet] = None
        if self.use_ema:
            self.ema_model = PolicyValueNet().to(device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        self.amp = device.startswith('cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=scheduler_Tmax)
        self._compiled = False
        if compile_model and device.startswith("cuda"):
            if not _triton_available():
                print("[WARN] torch.compile disabled (no Triton/CUDA compat). Running eager.")
            else:
                try:
                    import torch._dynamo as _dynamo
                    _dynamo.config.suppress_errors = True
                    self.model = torch.compile(self.model, backend=compile_backend)
                    self._compiled = True
                    print(f"[INFO] torch.compile enabled (backend={compile_backend})")
                except Exception as e:
                    print(f"[WARN] torch.compile failed: {e}. Fallback eager.")
                    self._compiled = False

    def soft_policy_loss(self, logits: torch.Tensor, target_pi: torch.Tensor, return_log_probs: bool = False):
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(target_pi * log_probs).sum(dim=1)
        if return_log_probs:
            return loss, log_probs
        return loss

    def _batch_augment_in_learner(self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray):
        if obs.shape[0] == 0:
            return obs, pi, z

        flip_indices = np.random.rand(obs.shape[0]) < 0.5
        if flip_indices.any():
            obs[flip_indices] = obs[flip_indices][:, :, :, ::-1]
            pi[flip_indices]  = pi[flip_indices][:, ::-1]

        return obs, pi, z

    def train_step(self, batch_size=512):
        if len(self.buffer) == 0:
            return 0.0, 0.0, 0.0

        total_loss = total_pl = total_vl = 0.0
        for _ in range(self.updates_per_iter):
            idxs, obs, pi, z, weights = self.buffer.sample(batch_size)
            if obs.shape[0] == 0:
                continue

            obs, pi, z = self._batch_augment_in_learner(obs, pi, z)

            obs_t = torch.from_numpy(obs).to(self.device)
            pi_t  = torch.from_numpy(pi).to(self.device)
            z_t   = torch.from_numpy(z).to(self.device)
            w_t   = torch.from_numpy(weights).to(self.device)

            # MODIFICATO: Aggiornata la sintassi di autocast secondo le nuove linee guida di PyTorch.
            with torch.amp.autocast(device_type=self.device, dtype=torch.float16, enabled=self.amp):
                logits, v = self.model(obs_t)
                per_sample_pl, log_probs = self.soft_policy_loss(logits, pi_t, return_log_probs=True)
                if self.smooth_value_loss:
                    per_sample_vl = F.smooth_l1_loss(v.squeeze(-1), z_t, reduction='none')
                else:
                    per_sample_vl = (v.squeeze(-1) - z_t) ** 2

                td_errors = (v.detach().squeeze(-1) - z_t).cpu().numpy()
                self.buffer.update_priorities(idxs, td_errors)

                policy_loss = (w_t * per_sample_pl).mean()
                value_loss  = (w_t * per_sample_vl).mean()

                probs = torch.exp(log_probs)
                entropy = -(probs * log_probs).sum(dim=1).mean()
                loss = policy_loss + self.value_loss_weight * value_loss - self.policy_entropy_weight * entropy

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self._update_ema()
            self.scheduler.step()

            total_loss += float(loss.item())
            total_pl   += float(policy_loss.item())
            total_vl   += float(value_loss.item())

        k = float(self.updates_per_iter)
        return total_loss/k, total_pl/k, total_vl/k

    def _update_ema(self):
        if not self.use_ema or self.ema_model is None:
            return
        with torch.no_grad():
            ema_params = dict(self.ema_model.named_parameters())
            for name, param in self.model.named_parameters():
                ema_params[name].data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
            ema_buffers = dict(self.ema_model.named_buffers())
            for name, buf in self.model.named_buffers():
                ema_buffers[name].data.copy_(buf.data)

    @staticmethod
    def _state_dict_on_cpu(model: nn.Module):
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def get_state_dict(self, ema: bool = True):
        model = self.ema_model if (ema and self.ema_model is not None) else self.model
        return self._state_dict_on_cpu(model)

    def save(self, path: str):
        payload = {
            "model": self._state_dict_on_cpu(self.model),
            "opt": self.opt.state_dict(),
            "sched": self.scheduler.state_dict(),
        }
        if self.use_ema and self.ema_model is not None:
            payload["ema"] = self._state_dict_on_cpu(self.ema_model)
        torch.save(payload, path)
    
    def load(self, path: str, fine_tuning: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

        if not fine_tuning:
            if "opt" in ckpt:
                try: self.opt.load_state_dict(ckpt["opt"])
                except Exception: pass
            if "sched" in ckpt:
                try: self.scheduler.load_state_dict(ckpt["sched"])
                except Exception: pass
        else:
            print("[INFO] Fine-tuning mode: Optimizer and scheduler states were NOT loaded.")
