# train.py --compile  --use_gumbel_actions  --dynamic_gumbel_k 
import argparse, contextlib, os, shutil, time, torch
from multiprocessing import Process, Queue, set_start_method, freeze_support
from queue import Empty
from typing import Dict
import numpy as np

from queue import Empty, Full # <-- Add Full to the import
import traceback # <-- Import traceback for better error logging
from connect4 import C4State, ROWS, COLS
from model import PolicyValueNet
from trainer import Trainer, self_play_worker, Sample 
from mcts_gumbel import GumbelMCTS

def reanalysis_worker(params_q: Queue, in_q: Queue, out_q: Queue, device: str, seed: int):
    # This initial setup part is fine
    import os as _os, torch as _torch, numpy as _np, random as _random
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _torch.set_num_threads(1)

    from model import PolicyValueNet as _PVN

    _torch.manual_seed(seed); _np.random.seed(seed); _random.seed(seed)
    model = _PVN().to(device).eval()

    def maybe_sync():
        try:
            sd = params_q.get_nowait()
            model.load_state_dict(sd)
            model.eval()
        except Empty: # Be specific about the exception
            pass

    @_torch.no_grad()
    def reanalyze_batch(obs_batch, z_batch):
        if obs_batch.shape[0] == 0:
            return []

        obs_t = _torch.from_numpy(obs_batch).to(device)
        logits, _ = model(obs_t)

        cur = (obs_t[:, 0] > 0.5)
        opp = (obs_t[:, 1] > 0.5)
        occ = (cur | opp)
        legal_mask = ~occ[:, 0, :]

        masked_logits = logits.masked_fill(~legal_mask, -1e9)
        new_pi = _torch.softmax(masked_logits, dim=1).detach().cpu().numpy().astype(_np.float32)

        return [Sample(obs=obs_batch[i], pi=new_pi[i], z=float(z_batch[i]))
                for i in range(obs_batch.shape[0])]

    maybe_sync()
    last_sync = time.time()

    while True:
        if time.time() - last_sync > 30:
            maybe_sync()
            last_sync = time.time()
        
        try:
            # Get a batch to process
            obs_b, z_b = in_q.get(timeout=5.0)
            
            # Reanalyze it
            updated_samples = reanalyze_batch(obs_b, z_b)
            
            # Put results back into the output queue without blocking
            for s in updated_samples:
                try:
                    # Use put_nowait to avoid blocking
                    out_q.put_nowait(s)
                except Full:
                    # If the queue is full, we just drop the sample and move on.
                    # This prevents the deadlock.
                    pass # Or you could add a counter for dropped samples
                    
        except Empty:
            # This is normal, just means no work was available
            continue
        except Exception:
            # Catch other unexpected errors and print them instead of failing silently
            print("\n--- ERROR IN REANALYSIS WORKER ---")
            traceback.print_exc()
            print("----------------------------------\n")
            continue

def drain_queue(out_q: Queue, trainer: Trainer, max_items: int = 5000) -> int:
    n = 0
    while n < max_items:
        try:
            sample = out_q.get_nowait()
            trainer.buffer.push(sample)
            n += 1
        except Empty:
            break
    return n

def cleanup_pycache(root="."):
    for path, dirs, files in os.walk(root):
        for d in dirs:
            if d == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(path, d))
                    print(f"[cleanup] removed {os.path.join(path, d)}")
                except Exception:
                    pass

def evaluate_agent(trainer: Trainer, games: int, sims: int, temperature: float,
                   baseline: str = "random", baseline_depth: int = 4) -> Dict[str, float]:
    if games <= 0:
        return {"wins": 0, "losses": 0, "draws": 0, "avg_len": 0.0,
                "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

    import inspect

    device = trainer.device
    eval_model = PolicyValueNet().to(device)
    eval_model.load_state_dict(trainer.get_state_dict(ema=True))
    eval_model.eval()

    params = inspect.signature(GumbelMCTS.__init__).parameters
    kw = dict(model=eval_model, num_simulations=sims, device=device)
    opt = {
        "root_dirichlet_alpha": None, "root_dirichlet_frac": 0.0, "c_puct": 1.0,
        "c_init": 1.25, "c_base": 19652.0, "fpu_reduction": 0.2, "gumbel_scale": 0.0,
    }
    for k, v in opt.items():
        if k in params: kw[k] = v

    baseline_depth = max(1, int(baseline_depth))
    stats = {"wins": 0, "losses": 0, "draws": 0}
    lengths = []
    baseline = baseline.lower()
    rng = np.random.default_rng()

    def heuristic_score(state: C4State, player: int) -> int:
        board = state.board; opp = -player; score = 0
        center_col = [board[r][COLS // 2] for r in range(ROWS)]; score += center_col.count(player) * 3
        def score_window(window):
            val=0; p_count=window.count(player); o_count=window.count(opp); z_count=window.count(0)
            if p_count == 4: val += 100
            elif p_count == 3 and z_count == 1: val += 5
            elif p_count == 2 and z_count == 2: val += 2
            if o_count == 3 and z_count == 1: val -= 4
            if o_count == 4: val -= 100
            return val
        for r in range(ROWS):
            row = board[r]
            for c in range(COLS - 3): score += score_window(row[c:c + 4])
        for c in range(COLS):
            col = [board[r][c] for r in range(ROWS)]
            for r in range(ROWS - 3): score += score_window(col[r:r + 4])
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score += score_window(window)
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                window = [board[r + i][c - i] for i in range(4)]
                score += score_window(window)
        return score

    def minimax(state: C4State, depth: int, alpha: float, beta: float, maximizing: bool, player: int, trans) -> float:
        key = (state.hash_key(), depth, maximizing)
        if key in trans: return trans[key]
        term, outcome = state.terminal()
        if term:
            if outcome == player: return 1000.0 + depth
            if outcome == -player: return -1000.0 - depth
            return 0.0
        if depth == 0: return float(heuristic_score(state, player))
        legal = state.legal_actions()
        if maximizing:
            value = -1e9
            for a in legal:
                child = state.apply(a)
                value = max(value, minimax(child, depth - 1, alpha, beta, False, player, trans))
                alpha = max(alpha, value)
                if alpha >= beta: break
            trans[key] = value; return value
        value = 1e9
        for a in legal:
            child = state.apply(a)
            value = min(value, minimax(child, depth - 1, alpha, beta, True, player, trans))
            beta = min(beta, value)
            if beta <= alpha: break
        trans[key] = value; return value

    def choose_minimax_action(state: C4State) -> int:
        legal = state.legal_actions(); rng.shuffle(legal); player = state.player
        best_score, best_moves = -1e9, []; trans = {}
        for action in legal:
            next_state = state.apply(action)
            score = minimax(next_state, baseline_depth - 1, -1e9, 1e9, False, player, trans)
            if score > best_score + 1e-6:
                best_score, best_moves = score, [action]
            elif abs(score - best_score) <= 1e-6:
                best_moves.append(action)
        return int(rng.choice(best_moves)) if best_moves else int(rng.choice(legal))

    for g in range(games):
        agent_player = 1 if (g % 2 == 0) else -1
        state = C4State.initial()
        mcts = GumbelMCTS(**kw)
        while True:
            legal = state.legal_actions(); action = -1
            if not legal: break
            if state.player == agent_player:
                pi, _ = mcts.run(state, temperature=temperature)
                action = max(legal, key=lambda a: pi[a])
                if pi[action] <= 0.0: action = int(rng.choice(legal))
            else:
                if baseline == "random": action = int(rng.choice(legal))
                elif baseline == "minimax": action = choose_minimax_action(state)
                else: raise ValueError(f"Unsupported baseline: {baseline}")
            state = state.apply(action); mcts.update_root(action)
            term, outcome = state.terminal()
            if term:
                lengths.append(state.ply)
                if outcome == agent_player: stats["wins"] += 1
                elif outcome == -agent_player: stats["losses"] += 1
                else: stats["draws"] += 1
                break
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    total = float(games)
    stats["avg_len"] = avg_len; stats["win_rate"] = stats["wins"] / total
    stats["draw_rate"] = stats["draws"] / total; stats["loss_rate"] = stats["losses"] / total
    return stats

def main_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=250_000, help='approx total samples to collect')
    parser.add_argument('--num_workers', type=int, default=14, help='i7-8700: 6 is ok; reduce if needed')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--updates_per_iter', type=int, default=16)
    parser.add_argument('--sims', type=int, default=2000, help='MCTS simulations per move')
    parser.add_argument('--gumbel_scale', type=float, default=0.1)
    parser.add_argument('--device', type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument('--save', type=str, default="c4_advanced.pt")
    parser.add_argument('--load', type=str, default="C:\\Users\\cm03696\\Desktop\\connect 4\\c4_advanced.pt", help='checkpoint to resume from')
    parser.add_argument('--compile', action="store_true", help="try torch.compile (Ampere+ recommended)")
    parser.add_argument('--no_cleanup', action="store_true", help="skip __pycache__ cleanup")
    parser.add_argument('--eval_every', type=float, default=600.0, help='seconds between evaluations (0 disables)')
    parser.add_argument('--eval_games', type=int, default=5, help='number of games in each evaluation batch')
    parser.add_argument('--eval_sims', type=int, default=1600, help='MCTS simulations per move during evaluation')
    parser.add_argument('--eval_temperature', type=float, default=1e-3, help='temperature for evaluation moves')
    parser.add_argument('--eval_baseline', type=str, default="minimax", help='baseline opponent (random, minimax)')
    parser.add_argument('--eval_baseline_depth', type=int, default=4, help='search depth for minimax baseline')
    parser.add_argument('--eval_on_start', action="store_true", help='run an evaluation before training loop')
    
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha (prioritization exponent)')
    parser.add_argument('--per_beta', type=float, default=0.4, help='PER beta (importance sampling exponent)')
    parser.add_argument('--use_gumbel_actions', action='store_true', help='Use Gumbel-based action selection in self-play')
    parser.add_argument('--dynamic_gumbel_k',action='store_true', help='Use dynamic k (3->2->1) based on game phase for Gumbel selection.')
    parser.add_argument('--num_reanalysis_workers', type=int, default=2, help='Number of workers for re-analyzing old data (0 to disable)')
    
    args = parser.parse_args()

    if not args.no_cleanup:
        cleanup_pycache(".")
    
    trainer = Trainer(device=args.device, sims=args.sims, gumbel_scale=args.gumbel_scale,
                      updates_per_iter=args.updates_per_iter, compile_model=args.compile,
                      per_alpha=args.per_alpha, per_beta=args.per_beta)
    if args.load and os.path.isfile(args.load):
        trainer.load(args.load, fine_tuning=True)
        print(f"[resume] loaded {args.load}")

    params_q = Queue(maxsize=args.num_workers + args.num_reanalysis_workers)
    out_q = Queue(maxsize=200_000)
    reanalysis_in_q = Queue(maxsize=128) if args.num_reanalysis_workers > 0 else None

    workers = []
    for i in range(args.num_workers):
        p = Process(target=self_play_worker, args=(params_q, out_q, trainer.device, args.sims, 
                                                   args.gumbel_scale, 1234 + i, 
                                                   args.use_gumbel_actions, args.dynamic_gumbel_k))
        p.start(); workers.append(p)

    if reanalysis_in_q is not None:
        for i in range(args.num_reanalysis_workers):
            p = Process(target=reanalysis_worker, args=(params_q, reanalysis_in_q, out_q, 
                                                        trainer.device, 5678 + i))
            p.daemon = True
            p.start(); workers.append(p)

    params_q.put(trainer.get_state_dict())
    last_sync = time.time(); last_save = time.time()
    last_log  = time.time(); last_eval = time.time()
    last_reanalysis_dispatch = time.time()
    collected, train_steps = 0, 0
    last_losses = (0.0, 0.0, 0.0)

    if args.eval_on_start:
        stats = evaluate_agent(trainer, args.eval_games, args.eval_sims,
                               args.eval_temperature, args.eval_baseline, args.eval_baseline_depth)
        print(f"[eval] init games={args.eval_games} W:{stats['wins']} D:{stats['draws']} L:{stats['losses']} "
              f"wr={stats['win_rate']*100:.1f}% avg_len={stats['avg_len']:.1f}")
        
        
        
        last_eval = time.time()

    try:
        while collected < args.episodes:
            collected += drain_queue(out_q, trainer, max_items=10000)

            if len(trainer.buffer) >= args.batch_size:
                l = trainer.train_step(batch_size=args.batch_size)
                last_losses = l; train_steps += 1

            if time.time() - last_sync > 15:
                while not params_q.empty():
                    try: params_q.get_nowait()
                    except Empty: break
                for _ in range(args.num_workers + args.num_reanalysis_workers):
                    try: params_q.put_nowait(trainer.get_state_dict())
                    except Exception: pass
                last_sync = time.time()
            
            '''if reanalysis_in_q is not None and time.time() - last_reanalysis_dispatch > 5.0:
                if len(trainer.buffer) > 2 * args.batch_size and not reanalysis_in_q.full():
                    # MODIFICA: Campiona un lotto molto più piccolo per la ri-analisi
                    reanalysis_batch_size = 512
                    _, obs, _, z, _ = trainer.buffer.sample(reanalysis_batch_size)
                    reanalysis_in_q.put((obs, z))
                    last_reanalysis_dispatch = time.time()
'''

            if reanalysis_in_q is not None and (time.time() - last_reanalysis_dispatch) > 25.0:
                if len(trainer.buffer) > 2 * args.batch_size:
                    reanalysis_batch_size = 512  # 64–128 va bene; parto da 128
                    _, obs, _, z, _ = trainer.buffer.sample(reanalysis_batch_size)
                    try:
                        reanalysis_in_q.put_nowait((obs, z))   # non bloccare
                        last_reanalysis_dispatch = time.time() # aggiorna il timer SOLO se inviato
                    except Full:
                        pass

            if args.eval_every > 0 and (time.time() - last_eval) >= args.eval_every:
                collected += drain_queue(out_q, trainer, max_items=20000)
                stats = evaluate_agent(trainer, args.eval_games, args.eval_sims,
                                       args.eval_temperature, args.eval_baseline, args.eval_baseline_depth)
                print(f"[eval] samples={collected} games={args.eval_games} "
                      f"W:{stats['wins']} D:{stats['draws']} L:{stats['losses']} "
                      f"wr={stats['win_rate']*100:.1f}% avg_len={stats['avg_len']:.1f}")
                

                
                last_eval = time.time()

            if time.time() - last_log > 5:
                l, pl, vl = last_losses
                ra_q_size = reanalysis_in_q.qsize() if reanalysis_in_q else 0
                print(f"[status] samples={collected} buffer={len(trainer.buffer)} "
                      f"train_steps={train_steps} loss={l:.4f} pl={pl:.4f} vl={vl:.4f} "
                      f"lr={trainer.opt.param_groups[0]['lr']:.2e} re-an_q={ra_q_size}")
                last_log = time.time()
                
            if time.time() - last_save > 180:
                trainer.save(args.save)
                last_save = time.time()

    except KeyboardInterrupt:
        print("Interrupted by user. Saving checkpoint...")

    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("!!! ERRORE NON GESTITO RILEVATO !!!")
        print(f"Tipo di Errore: {type(e).__name__}")
        print(f"Messaggio: {e}")
        print("\n--- TRACEBACK ---")
        traceback.print_exc() # Stampa esattamente dove si è rotto il codice
        print("="*60 + "\n")
    
    finally:
        try:
            trainer.save(args.save)
            print(f"Saved to {args.save}; collected {collected} samples; buffer={len(trainer.buffer)}")
        except Exception as exc:
            print(f"[error] failed to save checkpoint {args.save}: {exc}")
        for p in workers:
            if p.is_alive(): p.terminate()
            p.join(timeout=5)
            if p.is_alive(): print(f"[warn] worker {p.pid} did not terminate cleanly")
        queues = [params_q, out_q]
        if reanalysis_in_q: queues.append(reanalysis_in_q)
        for q in queues:
            with contextlib.suppress(Exception): q.close()
            with contextlib.suppress(Exception): q.join_thread()

if __name__ == '__main__':
    try: set_start_method('spawn')
    except RuntimeError: pass
    freeze_support()
    main_train()