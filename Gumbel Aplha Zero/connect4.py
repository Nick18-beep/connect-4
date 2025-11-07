
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
# connect4.py
from numba import njit

# Board geometry
ROWS, COLS, IN_A_ROW = 6, 7, 4
H, W = ROWS, COLS

# Bitboard mapping: 7 bits per column (6 playable + 1 sentinel).
# Bit index for (row r from bottom, col c): c*(H+1) + r
SHIFT_V  = 1          # vertical
SHIFT_H  = H + 1      # horizontal (7)
SHIFT_D1 = H          # diagonal / (6)
SHIFT_D2 = H + 2      # diagonal \ (8)

COL_MASKS    = [(((1 << H) - 1) << (c * (H + 1))) for c in range(COLS)]
BOTTOM_MASKS = [(1 << (c * (H + 1))) for c in range(COLS)]
TOP_MASKS    = [(1 << (c * (H + 1) + (H - 1))) for c in range(COLS)]


@njit(cache=True, inline='always')
def has_won_numba(bb: int) -> bool:
    """Versione compilata di _has_won."""
    m = bb & (bb >> SHIFT_V)
    if (m & (m >> (2 * SHIFT_V))) != 0: return True
    m = bb & (bb >> SHIFT_H)
    if (m & (m >> (2 * SHIFT_H))) != 0: return True
    m = bb & (bb >> SHIFT_D1)
    if (m & (m >> (2 * SHIFT_D1))) != 0: return True
    m = bb & (bb >> SHIFT_D2)
    if (m & (m >> (2 * SHIFT_D2))) != 0: return True
    return False


class C4State:
    """Connect-4 state using bitboards (fast).
    bb[0] -> player +1 stones, bb[1] -> player -1 stones.
    player -> side to move (+1 or -1).
    """
    __slots__ = ("bb", "player", "last_move_bit", "ply")

    def __init__(self, bb0: int = 0, bb1: int = 0, player: int = 1, last_move_bit: Optional[int] = None, ply: int = 0):
        self.bb = [int(bb0), int(bb1)]
        self.player = 1 if player >= 0 else -1
        self.last_move_bit = last_move_bit  # int mask or None
        self.ply = int(ply)

    @staticmethod
    def initial() -> "C4State":
        return C4State()

    def clone(self) -> "C4State":
        return C4State(self.bb[0], self.bb[1], self.player, self.last_move_bit, self.ply)

    @property
    def board(self) -> List[List[int]]:
        """Top-to-bottom 6x7 grid view (for printing / legacy compatibility)."""
        grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        for c in range(COLS):
            base = c * (H + 1)
            for r_bot in range(H):  # 0 bottom .. H-1 top
                bit = 1 << (base + r_bot)
                r_top = H - 1 - r_bot
                if self.bb[0] & bit:
                    grid[r_top][c] = 1
                elif self.bb[1] & bit:
                    grid[r_top][c] = -1
        return grid

    def legal_actions(self) -> List[int]:
        """List of columns where the top playable cell is empty."""
        legal = []
        mask_all = self.bb[0] | self.bb[1]
        for c in range(COLS):
            if (mask_all & TOP_MASKS[c]) == 0:
                legal.append(c)
        return legal

    def _lowest_empty_bit(self, mask_all: int, col: int) -> int:
        """Return bitmask of lowest empty cell in column, or 0 if full."""
        return (mask_all + BOTTOM_MASKS[col]) & COL_MASKS[col]

    def apply(self, action: int) -> "C4State":
        """Drop a piece in column `action` and return the next state."""
        c = int(action)
        mask_all = self.bb[0] | self.bb[1]
        move_bit = self._lowest_empty_bit(mask_all, c)
        if move_bit == 0:
            raise ValueError(f"Illegal move on full column {c}")
        next_state = self.clone()
        idx = 0 if self.player == 1 else 1
        next_state.bb[idx] |= move_bit
        next_state.player = -self.player
        next_state.last_move_bit = move_bit
        next_state.ply += 1
        return next_state

    def _has_won(self, bb: int) -> bool:
        return has_won_numba(bb)

    def terminal(self) -> Tuple[bool, Optional[int]]:
        """Return (done, winner) where winner in {+1, -1, 0} or None if not done."""
        last_idx = 0 if self.player == -1 else 1  # side who just moved
        if self.last_move_bit is not None:
            if self._has_won(self.bb[last_idx]):
                return True, -self.player
        # check draw
        mask_all = self.bb[0] | self.bb[1]
        for c in range(COLS):
            if (mask_all & TOP_MASKS[c]) == 0:
                return False, None
        return True, 0

    
    def to_planes(self) -> np.ndarray:
        """Return (C,H,W) float32 planes: [cur, opp, to_play, last_move]."""
        cur_idx = 0 if self.player == 1 else 1
        opp_idx = 1 - cur_idx
        cur = np.zeros((ROWS, COLS), dtype=np.float32)
        opp = np.zeros((ROWS, COLS), dtype=np.float32)
        for c in range(COLS):
            base = c * (H + 1)
            for r_bot in range(H):
                bit = 1 << (base + r_bot)
                r_top = H - 1 - r_bot
                if self.bb[cur_idx] & bit:
                    cur[r_top, c] = 1.0
                elif self.bb[opp_idx] & bit:
                    opp[r_top, c] = 1.0

        to_play = np.full((ROWS, COLS), 1.0 if self.player == 1 else 0.0, dtype=np.float32)
        last = np.zeros((ROWS, COLS), dtype=np.float32)
        if self.last_move_bit:
            pos = (self.last_move_bit.bit_length() - 1)
            c = pos // (H + 1)
            r_bot = pos % (H + 1)
            if r_bot < H:
                r_top = H - 1 - r_bot
                last[r_top, c] = 1.0
        return np.stack([cur, opp, to_play, last], axis=0).astype(np.float32)

    def hash_key(self) -> Tuple[int, int, int]:
        return (self.bb[0], self.bb[1], self.player)
