# infer.py

import argparse
import torch
from typing import List, Tuple

from connect4 import C4State, ROWS, COLS
from model import PolicyValueNet
from mcts_gumbel import GumbelMCTS

SYMBOLS = {0: '.', 1: 'X', -1: 'O'}

def print_board(state: C4State) -> None:
    grid = state.board
    for r in range(ROWS):
        row = ' '.join(SYMBOLS[grid[r][c]] for c in range(COLS))
        print(row)
    print(' '.join(map(str, range(COLS))))
    print("-" * (COLS * 2 - 1))

def format_topk(pi, k=3):
    idx = sorted(range(len(pi)), key=lambda i: pi[i], reverse=True)[:k]
    return ', '.join(f"{i}:{pi[i]:.2f}" for i in idx)

def ask_human_move(legal: List[int]) -> int:
    while True:
        try:
            col = int(input(f"Scegli colonna {legal}: "))
            if col in legal:
                return col
            print("Mossa illegale.")
        except Exception:
            print("Input non valido.")

def main_infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, required=False, default=r"C:\Users\cm03696\Desktop\connect 4\c4_advanced.pt")
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    parser.add_argument('--sims', type=int, default=5000)
    parser.add_argument('--temp', type=float, default=1e-6)
    parser.add_argument('--gumbel_scale', type=float, default=0)
    parser.add_argument('--root_noise_alpha', type=float, default=0.1)
    parser.add_argument('--root_noise_frac', type=float, default=0)
    parser.add_argument('--human_first', action='store_true')
    parser.add_argument('--show_topk', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu'
    model = PolicyValueNet().to(device)
    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        state_dict = ckpt.get('ema', ckpt.get('model', ckpt))
        model.load_state_dict(state_dict)
    model.eval()

    mcts = GumbelMCTS(model, num_simulations=args.sims, device=device,
                      root_dirichlet_alpha=args.root_noise_alpha,
                      root_dirichlet_frac=args.root_noise_frac,
                      c_puct=1.0, c_init=1.25, c_base=19652.0, fpu_reduction=0.2,
                      gumbel_scale=args.gumbel_scale)

    state = C4State.initial()
    print_board(state)

    while True:
        term, outcome = state.terminal()
        if term:
            if outcome == 0:
                print("\n== Patta ==")
            else:
                print(f"\n== Vince {'X' if outcome==1 else 'O'} ==")
            break

        move = -1
        if (args.human_first and state.player == 1) or ((not args.human_first) and state.player == -1):
            legal = state.legal_actions()
            move = ask_human_move(legal)
        else:
            print("\nTurno dell'agente...")
            pi, _ = mcts.run(state, temperature=args.temp)
            legal = state.legal_actions()
            if not legal:
                break
            move = max(legal, key=lambda a: pi[a])
            
            if args.show_topk > 0:
                print(f"[Agent] top{args.show_topk}: {format_topk(pi, args.show_topk)}")
            print(f"[Agent] gioca colonna: {move}")

        state = state.apply(move)
        mcts.update_root(move)
        
        print_board(state)

if __name__ == '__main__':
    main_infer()