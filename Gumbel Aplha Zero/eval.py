# eval_contro_minimax.py

import argparse
import torch
import time
import os
import sys
import numpy as np
import inspect
from typing import Dict, List, Tuple

# Assicura che i moduli importati non usino troppi thread
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Importa i componenti necessari dai tuoi file
try:
    from trainer import Trainer
    from connect4 import C4State, ROWS, COLS
    from model import PolicyValueNet
    from mcts_gumbel import GumbelMCTS
except ImportError as e:
    print(f"Errore di importazione: {e}", file=sys.stderr)
    print("Assicurati che 'eval_contro_minimax.py' sia nella stessa cartella di 'trainer.py', 'model.py', 'connect4.py' e 'mcts_gumbel.py'.", file=sys.stderr)
    sys.exit(1)

# =============================================================================
# FUNZIONE DI VALUTAZIONE (MODIFICATA SENZA STAMPE INTERMEDIE)
# =============================================================================

def evaluate_agent(trainer: Trainer, games: int, sims: int, temperature: float,
                   baseline: str = "random", baseline_depth: int = 4) -> Dict[str, float]:
    """
    Esegue N partite tra l'agente e un baseline.
    Questa versione non stampa output durante le partite, solo alla fine.
    """
    if games <= 0:
        return {"wins": 0, "losses": 0, "draws": 0, "avg_len": 0.0,
                "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0}

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

    # --- Funzioni interne per Minimax ---
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

    # --- Ciclo di gioco ---
    for g in range(games):
        agent_player = 1 if (g % 2 == 0) else -1
        state = C4State.initial()
        mcts = GumbelMCTS(**kw)
        
        # RIMOSSA: Stampa di inizio partita

        while True:
            legal = state.legal_actions(); action = -1
            if not legal: break
            
            if state.player == agent_player:
                # Turno dell'Agente MCTS
                pi, _ = mcts.run(state, temperature=temperature)
                action = max(legal, key=lambda a: pi[a])
                if pi[action] <= 0.0: action = int(rng.choice(legal))
            else:
                # Turno del Baseline (Minimax o Random)
                if baseline == "random": 
                    action = int(rng.choice(legal))
                elif baseline == "minimax": 
                    action = choose_minimax_action(state)
                else: 
                    raise ValueError(f"Unsupported baseline: {baseline}")
            
            # RIMOSSA: Stampa del tempo per mossa

            state = state.apply(action); mcts.update_root(action)
            term, outcome = state.terminal()
            if term:
                lengths.append(state.ply)
                if outcome == agent_player: 
                    stats["wins"] += 1
                elif outcome == -agent_player: 
                    stats["losses"] += 1
                else: 
                    stats["draws"] += 1
                
                # RIMOSSA: Stampa di fine partita (Vittoria/Sconfitta/Patta)
                break
                
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    total = float(games)
    stats["avg_len"] = avg_len; stats["win_rate"] = stats["wins"] / total
    stats["draw_rate"] = stats["draws"] / total; stats["loss_rate"] = stats["losses"] / total
    return stats


# =============================================================================
# FUNZIONE MAIN (PARAMETRI COME DA ULTIMA RICHIESTA)
# =============================================================================

def main_evaluation():
    parser = argparse.ArgumentParser(description="Valuta un modello MCTS contro un'AI Minimax per N partite.")
    
    parser.add_argument(
        '--load', 
        type=str, 
        default="C:\\Users\\cm03696\\Desktop\\connect 4\\c4_advanced.pt", 
        help="Percorso del checkpoint del modello (.pt) da valutare."
    )
    parser.add_argument(
        '--games', 
        type=int, 
        default=1, 
        help="Numero (N) di partite totali da giocare (verranno alternate le partenze)."
    )
    parser.add_argument(
        '--sims', 
        type=int, 
        default=5000, 
        help="Numero di simulazioni MCTS per mossa per il tuo agente."
    )
    parser.add_argument(
        '--depth', 
        type=int, 
        default=5, 
        help="Profondità di ricerca (ply) per l'avversario Minimax."
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto', 
        choices=['auto', 'cpu', 'cuda'],
        help="Device su cui far girare il modello MCTS ('auto', 'cpu', 'cuda')."
    )
    
    args = parser.parse_args()

    if args.games <= 0:
        print("Errore: Il numero di partite deve essere maggiore di 0.", file=sys.stderr)
        return

    # 1. Determina il device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"Attenzione: Hai richiesto 'cuda' ma non è disponibile. Verrà usato 'cpu'.")
        device = 'cpu'

    print(f"--- Avvio Valutazione ---")
    print(f"Device selezionato: {device}")
    print(f"Modello da caricare: {args.load}")

    # 2. Crea un'istanza di Trainer
    try:
        trainer = Trainer(device=device, compile_model=False)
    except Exception as e:
        print(f"Errore nella creazione del Trainer: {e}", file=sys.stderr)
        return

    # 3. Carica il checkpoint nel Trainer
    if not os.path.isfile(args.load):
        print(f"Errore: File checkpoint non trovato: {args.load}", file=sys.stderr)
        return
        
    try:
        print(f"Caricamento del modello...")
        trainer.load(args.load, fine_tuning=True)
    except Exception as e:
        print(f"Errore irreversibile during il caricamento del modello: {e}", file=sys.stderr)
        return

    print(f"Modello caricato con successo.")
    print("-------------------------")
    print(f"Partite da giocare: {args.games}")
    print(f"Agente (MCTS): {args.sims} simulazioni/mossa")
    print(f"Avversario (Minimax): profondità {args.depth}")
    print("-------------------------")
    print(f"Inizio partite... (output mostrato solo alla fine)") # Modificata

    # 4. Esegui la valutazione
    start_time = time.time()
    
    stats = evaluate_agent(
        trainer=trainer,
        games=args.games,
        sims=args.sims,
        temperature=1e-6,
        baseline="minimax",
        baseline_depth=args.depth
    )
    
    end_time = time.time()

    # 5. Stampa le statistiche finali (QUESTA PARTE È MANTENUTA)
    total_time = end_time - start_time
    avg_time_per_game = total_time / args.games if args.games > 0 else 0

    print("\n--- Risultati Finali della Valutazione ---")
    print(f"Tempo totale: {total_time:.2f} secondi")
    print(f"Tempo medio per partita: {avg_time_per_game:.2f} secondi")
    print(f"Partite totali giocate: {args.games}")
    print(f"Durata media partita (ply): {stats['avg_len']:.1f}")
    print("------------------------------------------")
    print(f"Vittorie Agente (MCTS): {stats['wins']} \t({stats['win_rate']*100:.1f}%)")
    print(f"Sconfitte Agente (Minimax vince): {stats['losses']} \t({stats['loss_rate']*100:.1f}%)")
    print(f"Patte: {stats['draws']} \t({stats['draw_rate']*100:.1f}%)")
    print("------------------------------------------")

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    torch.multiprocessing.freeze_support()
    
    main_evaluation()