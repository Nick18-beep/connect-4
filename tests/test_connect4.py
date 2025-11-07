from __future__ import annotations

import numpy as np
import pytest

from connect4 import C4State, COLS

def _play_sequence(seq):
    state = C4State.initial()
    for move in seq:
        state = state.apply(move)
    return state


def test_initial_state_has_all_legal_moves():
    state = C4State.initial()
    assert state.player == 1
    assert state.legal_actions() == list(range(COLS))


def test_apply_alternates_player_and_updates_last_move():
    state = C4State.initial()
    next_state = state.apply(3)
    assert next_state.player == -1
    assert next_state.last_move_bit is not None
    # Column 3 should now be filled at bottom for player +1
    board = next_state.board
    bottom_row = board[-1]
    assert bottom_row[3] == 1


def test_apply_raises_on_full_column():
    state = C4State.initial()
    for _ in range(6):  # fill column 0
        state = state.apply(0)
    with pytest.raises(ValueError):
        state.apply(0)


def test_terminal_detects_vertical_win():
    state = _play_sequence([1, 2, 1, 2, 1, 2, 1])
    done, winner = state.terminal()
    assert done
    assert winner == 1


def test_terminal_detects_diagonal_win():
    # Diagonal from bottom-left to top-right for player +1
    state = _play_sequence([0, 1, 1, 2, 2, 3, 2, 3, 3, 4, 3])
    done, winner = state.terminal()
    assert done
    assert winner == 1


def test_terminal_detects_draw_on_full_board():
    draw_seq = [3, 6, 1, 3, 1, 3, 6, 4, 5, 5, 3, 4, 3, 1, 1, 0,
                4, 3, 0, 4, 6, 6, 2, 6, 1, 0, 5, 5, 2, 4, 5, 1,
                5, 6, 0, 0, 2, 2, 4, 2, 2, 0]
    state = _play_sequence(draw_seq)
    done, winner = state.terminal()
    assert done
    assert winner == 0


def test_to_planes_encodes_cur_opp_and_last_move():
    state = C4State.initial()
    state = state.apply(3)  # +1 bottom column 3
    state = state.apply(2)  # -1 bottom column 2
    planes = state.to_planes()
    assert planes.shape == (4, 6, 7)

    cur_plane = planes[0]
    opp_plane = planes[1]
    to_play = planes[2]
    last_move = planes[3]

    # Current player is +1, so to_play plane is ones
    assert np.allclose(to_play, 1.0)

    # Bottom row is index 5 due to top-oriented representation
    assert cur_plane[5, 3] == 1.0
    assert opp_plane[5, 2] == 1.0

    # Last move plane highlights column 2 bottom cell
    assert last_move[5, 2] == 1.0


def test_hash_key_is_stable_under_clone():
    state = C4State.initial()
    state = state.apply(0)
    key = state.hash_key()
    clone = state.clone()
    assert clone.hash_key() == key


def test_mask_all_updates_after_apply():
    state = C4State.initial()
    assert state.mask_all == 0
    state = state.apply(3)
    assert state.mask_all != 0
    clone = state.clone()
    assert clone.mask_all == state.mask_all
