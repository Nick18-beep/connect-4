# mcts_gumbel.py — Gumbel Top-k MCTS ottimizzato per Connect4
# Funzioni principali:
# - Gumbel Top-k al root (vera selezione Top-k sul prior perturbato)
# - Dirichlet noise al root
# - PUCT + FPU
# - Fast-path tattici: win-in-1, block-in-1
# - Penalty "avoid losing move" con decay per profondità
# - Virtual loss su visite per batching parallelo
# - LRU cache per terminal/win-in-1 (chiave robusta a last_move_bit=None)
# - Policy da visite stabile numericamente (softmax(log(visits)/T))

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Set
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn

from connect4 import (
    C4State,
    COLS,
    ROWS,
    COL_MASKS,
    BOTTOM_MASKS,
    TOP_MASKS,
    FULL_TOP_MASK,
    has_won_numba,
)

__version__ = "mcts_gumbel 3.2 (true-gumbel-topk + block-in-1 + virtual-loss + depth-decay + stable-temp)"

@lru_cache(maxsize=8192)
def _terminal_lookup(bb0: int, bb1: int, player: int, last_move_bit: int, mask_all: int) -> Tuple[bool, Optional[int]]:
    last_bb = bb0 if player == -1 else bb1
    if last_move_bit and has_won_numba(last_bb):
        return True, -player
    if (mask_all & FULL_TOP_MASK) == FULL_TOP_MASK:
        return True, 0
    return False, None


@lru_cache(maxsize=8192)
def _threats_lookup(cur_bb: int, mask_all: int) -> Tuple[int, ...]:
    threats: List[int] = []
    for a in range(COLS):
        if mask_all & TOP_MASKS[a]:
            continue
        move_bit = (mask_all + BOTTOM_MASKS[a]) & COL_MASKS[a]
        if move_bit == 0:
            continue
        if has_won_numba(cur_bb | move_bit):
            threats.append(a)
    return tuple(threats)


# =========================
#   Nodo dell'albero
# =========================
class Node:
    __slots__ = ("parent", "prior", "N", "W", "Q", "children", "to_play", "is_expanded")
    def __init__(self, prior: float, to_play: int, parent: Optional["Node"] = None):
        self.parent: Optional["Node"] = parent
        self.prior: float = float(prior)
        self.N: float = 0.0           # visite (float per virtual loss)
        self.W: float = 0.0           # somma dei valori
        self.Q: float = 0.0           # valore medio
        self.children: Dict[int, Node] = {}
        self.to_play: int = int(to_play)
        self.is_expanded: bool = False

# =========================
#   MCTS con Gumbel Top-k
# =========================
class GumbelMCTS:
    def __init__(
        self,
        model: nn.Module,
        num_simulations: int = 160,
        batch_size: int = 32,
        virtual_loss: float = 1.0,
        c_puct: float = 1.0,
        c_init: float = 1.25,
        c_base: float = 19652.0,
        root_dirichlet_alpha: Optional[float] = 1.0,
        root_dirichlet_frac: float = 0.25, #
        fpu_reduction: float = 0.2,
        gumbel_scale: float = 1.0,
        device: str = "cpu",
        cache_size: int = 4096,
        avoid_lose_depth: int = 3,     # applica penalty solo nei primi livelli
        avoid_lose_penalty: float = 1.0,  # Penalità base per 1 minaccia
        avoid_fork_penalty: float = 2.5,  # Penalità aumentata per >= 2 minacce
        rng_seed: Optional[int] = None,
    ):
        self.model = model
        self.num_simulations = int(num_simulations)
        self.batch_size = int(max(1, batch_size))
        self.virtual_loss = float(max(0.0, virtual_loss))
        self.c_puct = float(c_puct)
        self.c_init = float(c_init)
        self.c_base = float(c_base)
        self.root_dirichlet_alpha = float(root_dirichlet_alpha) if root_dirichlet_alpha is not None else None
        self.root_dirichlet_frac = float(root_dirichlet_frac)
        self.fpu_reduction = float(fpu_reduction)
        self.gumbel_scale = float(max(0.0, gumbel_scale))
        self.device = device

        self.root: Optional[Node] = None
        self._root_allowed_actions: Optional[Set[int]] = None

        # RNG & cache
        self.rng = np.random.default_rng(rng_seed)
        self._avoid_lose_depth = int(avoid_lose_depth)

        # <--- MODIFICATO: Salvataggio dei nuovi parametri di penalità ---
        self.avoid_lose_penalty = float(avoid_lose_penalty)
        self.avoid_fork_penalty = float(avoid_fork_penalty)

        # Modello in eval
        self.model.eval()

    # ---------- API pubblica ----------
    def get_root_q_values(self) -> Dict[int, float]:
        if self.root is None:
            return {}
        # Q dal punto di vista dell'avversario (utile per debug/plot)
        return {a: -ch.Q for a, ch in self.root.children.items()}

    def update_root(self, action: int):
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None

    # ---------- Tattiche & cache ----------
    def _terminal_cached(self, s: C4State) -> Tuple[bool, Optional[int]]:
        last_move_bit = s.last_move_bit or 0
        return _terminal_lookup(s.bb[0], s.bb[1], s.player, last_move_bit, s.mask_all)
    
    # <--- MODIFICATO: Nuova funzione _get_opponent_threats ---
    def _get_opponent_threats(self, s_with_opp_to_move: C4State) -> List[int]:
        """
        Trova TUTTE le mosse vincenti in 1 per il giocatore che muove nello stato 's'.
        """
        cur_idx = 0 if s_with_opp_to_move.player == 1 else 1
        cur_bb = s_with_opp_to_move.bb[cur_idx]
        threats = _threats_lookup(cur_bb, s_with_opp_to_move.mask_all)
        return list(threats)

    # <--- MODIFICATO: _win_in_one ora usa la nuova funzione ---
    def _win_in_one(self, s: C4State) -> Optional[int]:
        """
        Ritorna la *prima* mossa vincente per il giocatore corrente, o None.
        (Mantiene la compatibilità con il fast-path in run())
        """
        threats = self._get_opponent_threats(s) # 's' è lo stato dove muove 's.player'
        return threats[0] if threats else None

    # <--- MODIFICATO: _opponent_wins_in_one_cols ora usa la nuova funzione ---
    def _opponent_wins_in_one_cols(self, s_with_opp_to_move: C4State) -> List[int]:
        """Ritorna TUTTE le mosse vincenti per l'avversario."""
        return self._get_opponent_threats(s_with_opp_to_move)

    def _unsafe_for_current_player(self, next_state: C4State) -> int:
        """
        Controlla quante minacce ha l'avversario in 'next_state'.
        Ritorna: 0 (sicuro), 1 (minaccia singola), 2 (minaccia multipla/fork)
        """
        threats = self._get_opponent_threats(next_state)
        num_threats = len(threats)
        if num_threats == 0:
            return 0
        elif num_threats == 1:
            return 1
        else:
            return 2 # 2 o più minacce

    # ---------- Core ----------
    @torch.no_grad()
    def run(self, state: C4State, temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """Esegue MCTS e ritorna (policy[COLS], root_value)."""
        # --- Fast path: win-in-1 o block-in-1 (unica minaccia) ---
        legal = state.legal_actions()
        if legal:
            w = self._win_in_one(state)
            if w is not None:
                pi = np.zeros(COLS, dtype=np.float32); pi[w] = 1.0
                return pi, 1.0

            s_opp = C4State(state.bb[0], state.bb[1], -state.player, state.last_move_bit, state.ply)
            opp_wins = self._opponent_wins_in_one_cols(s_opp)
            if len(opp_wins) == 1 and opp_wins[0] in legal:
                b = opp_wins[0]
                pi = np.zeros(COLS, dtype=np.float32); pi[b] = 1.0
                return pi, 0.0

        # --- Root setup / espansione ---
        if self.root is None:
            self.root = Node(prior=1.0, to_play=state.player)

        if not self.root.is_expanded or not self.root.children:
            self._expand_and_eval_root(state)

        # --- Gumbel Top-k al root ---
        self._root_allowed_actions = None
        root = self.root
        if self.gumbel_scale > 0.0 and root and root.children:
            actions = np.fromiter(root.children.keys(), dtype=np.int64)
            priors = np.fromiter((root.children[a].prior for a in actions), dtype=np.float64)
            priors = np.clip(priors, 1e-8, 1.0)
            g = self.rng.gumbel(0.0, self.gumbel_scale, size=priors.shape[0])
            scores = np.log(priors) + g

            L = actions.shape[0]
            if L <= 3:
                top_k = L
            else:
                nsim = self.num_simulations
                top_k = 3 if nsim >= 800 else (2 if nsim >= 200 else 1)
            top_k = max(1, min(top_k, L))

            # Indici top-k
            top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            self._root_allowed_actions = set(int(a) for a in actions[top_idx])

        # --- Simulazioni batched ---
        sims = max(1, self.num_simulations)
        remaining = sims
        bs = self.batch_size
        vloss = self.virtual_loss

        torch.set_grad_enabled(False)

        while remaining > 0:
            batch = bs if remaining >= bs else remaining
            remaining -= batch

            pending_nodes: List[Node] = []
            pending_states: List[C4State] = []
            paths: List[List[Tuple[Node, int]]] = []

            # SELEZIONE
            for _ in range(batch):
                path, leaf, leaf_state = self._select(state)
                paths.append(path)
                pending_nodes.append(leaf)
                pending_states.append(leaf_state)

            if not pending_nodes:
                continue

            # EVAL (batch)
            obs = np.empty((len(pending_states), 4, ROWS, COLS), dtype=np.float32)
            for i, s in enumerate(pending_states):
                obs[i] = s.to_planes()
            obs_t = torch.from_numpy(obs).to(self.device)
            logits, values = self.model(obs_t)
            pol_np = torch.softmax(logits, dim=1).cpu().numpy()
            val_np = values.squeeze(-1).detach().cpu().numpy()

            # ESPANSIONE & BACKUP
            for i in range(len(pending_nodes)):
                leaf_node = pending_nodes[i]
                sim_state = pending_states[i]

                term, outcome = self._terminal_cached(sim_state)
                if term:
                    if outcome == 0:
                        leaf_v = 0.0
                    elif outcome == leaf_node.to_play:
                        leaf_v = 1.0
                    else:
                        leaf_v = -1.0
                else:
                    legal_i = sim_state.legal_actions()
                    pri = pol_np[i, legal_i].astype(np.float64, copy=False)
                    s = pri.sum()
                    if s <= 1e-12:
                        pri = np.full(len(legal_i), 1.0 / len(legal_i), dtype=np.float64)
                    else:
                        pri /= s
                    for idx, a in enumerate(legal_i):
                        if a not in leaf_node.children:
                            leaf_node.children[a] = Node(
                                prior=float(pri[idx]),
                                to_play=-leaf_node.to_play,
                                parent=leaf_node
                            )
                    leaf_node.is_expanded = True
                    leaf_v = float(val_np[i])

                # BACKUP (senza pop(0), rimuovi virtual loss del path)
                v = leaf_v
                path = paths[i]
                for node, a in reversed(path):
                    ch = node.children.get(a)
                    if ch is not None and vloss > 0.0:
                        ch.N = ch.N - vloss if ch.N > vloss else 0.0
                    node.N += 1.0
                    node.W += v
                    node.Q = node.W / node.N
                    v = -v

        # --- POLICY da visite (stabile numericamente) ---
        pi = np.zeros(COLS, dtype=np.float32)
        if root and root.children:
            legal = list(root.children.keys())
            visits = np.array([root.children[a].N for a in legal], dtype=np.float64)
            total_visits = float(visits.sum())

            if total_visits <= 0.0:
                pri = np.array([root.children[a].prior for a in legal], dtype=np.float64)
                s = float(pri.sum())
                if s > 0.0:
                    pri /= s
                else:
                    pri = np.full(len(legal), 1.0 / len(legal), dtype=np.float64)
                for idx, a in enumerate(legal):
                    pi[a] = float(pri[idx])
            else:
                T = float(temperature)
                if T <= 1e-8:
                    a_star = int(legal[int(np.argmax(visits))])
                    pi[a_star] = 1.0
                elif abs(T - 1.0) < 1e-6:
                    probs = visits / total_visits
                    for idx, a in enumerate(legal):
                        pi[a] = float(probs[idx])
                else:
                    eps = 1e-12
                    logits = np.log(visits + eps) / T
                    logits -= np.max(logits)  # stabilizzazione
                    probs = np.exp(logits)
                    s = float(probs.sum())
                    if not np.isfinite(s) or s <= 0.0:
                        probs = visits / total_visits
                    else:
                        probs /= s
                    for idx, a in enumerate(legal):
                        pi[a] = float(probs[idx])

        root_value = root.Q if root else 0.0
        return pi, float(root_value)

    # ---------- Internals ----------
    def _expand_and_eval_root(self, state: C4State):
        obs = torch.from_numpy(state.to_planes()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _values = self.model(obs)
            policy = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        legal = state.legal_actions()
        pri = policy[legal].astype(np.float64, copy=False)
        s = pri.sum()
        if s <= 1e-12:
            pri = np.full(len(legal), 1.0 / len(legal), dtype=np.float64)
        else:
            pri /= s

        # Dirichlet al root (se multi-legali)
        if self.root_dirichlet_alpha is not None and self.root_dirichlet_frac > 0.0 and len(legal) > 1:
            noise = self.rng.dirichlet([self.root_dirichlet_alpha] * len(legal))
            pri = (1.0 - self.root_dirichlet_frac) * pri + self.root_dirichlet_frac * noise

        root = self.root
        to_play_child = -root.to_play
        for idx, a in enumerate(legal):
            root.children[a] = Node(prior=float(pri[idx]), to_play=to_play_child, parent=root)
        root.is_expanded = True

    def _select(self, root_state: C4State) -> Tuple[List[Tuple[Node, int]], Node, C4State]:
        """Selezione con PUCT+FPU; al root rispetta l'eventuale Top-k; penalty avoid-lose
        solo su figli non visitati o nei primi livelli (riduce apply() costose)."""
        node = self.root
        state = root_state
        path: List[Tuple[Node, int]] = []

        vloss = self.virtual_loss
        c_init = self.c_init
        c_base = self.c_base
        c_puct = self.c_puct
        fpu_red = self.fpu_reduction

        depth = 0
        while node.is_expanded and node.children:
            # Candidati
            if depth == 0 and self._root_allowed_actions is not None:
                items = [(a, node.children[a]) for a in self._root_allowed_actions]
            else:
                items = list(node.children.items())

            Nn = node.N
            sqrt_N = math.sqrt(Nn if Nn > 1.0 else 1.0)
            pb_c = (math.log((Nn + c_base + 1.0) / c_base) + c_init) * c_puct

            best_a = -1
            best_score = -1e9
            do_avoid = depth < self._avoid_lose_depth

            
            # <--- MODIFICATO: Calcolo delle penalità basate sui parametri di __init__ ---
            depth_decay = 0.85 ** depth
            single_threat_penalty = self.avoid_lose_penalty * depth_decay
            double_threat_penalty = self.avoid_fork_penalty * depth_decay
            # --- Fine modifica ---

            for a, ch in items:
                n = ch.N
                # prior term
                prior_term = pb_c * ch.prior * (sqrt_N / (1.0 + n))
                # FPU per non visitati
                q = ch.Q if n > 0.0 else (node.Q - fpu_red)
                score = q + prior_term

                # Penalty "avoid losing move" su figli non visitati e early-depth
                # <--- MODIFICATO: Logica di penalità a più livelli ---
                # Penalty "avoid losing move" su figli non visitati e early-depth
                if do_avoid and (n < 1.0):
                    ns = state.apply(a)
                    # Controlla il numero di minacce avversarie create da questa mossa
                    threat_level = self._unsafe_for_current_player(ns)
                    
                    if threat_level >= 2:
                        # Penalità massima per fork/doppie minacce
                        score -= double_threat_penalty
                    elif threat_level == 1:
                        # Penalità standard per minaccia singola
                        score -= single_threat_penalty
                # --- Fine modifica ---

                if score > best_score:
                    best_score = score
                    best_a = a

            # Scendi
            path.append((node, best_a))
            child = node.children[best_a]
            if vloss > 0.0:
                child.N += vloss  # virtual visit solo sul figlio selezionato

            state = state.apply(best_a)
            node = child
            depth += 1

        return path, node, state
